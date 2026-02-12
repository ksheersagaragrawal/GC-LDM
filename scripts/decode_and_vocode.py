"""Reconstruct waveforms from saved AudioLDM VAE latents.

Pipeline:
1) Load latent tensors (`z`) and `genre_id` from data/processed_latents/*.pt
2) Decode with the AudioLDM VAE decoder to recover log-mel spectrograms
3) Either:
   - feed log-mel directly to HiFi-GAN, or
   - exponentiate log-mel and run Griffin-Lim fallback
"""

import argparse
import glob
import os
import json
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torchaudio
from diffusers import AutoencoderKL
from transformers import SpeechT5HifiGan


# -----------------------------
# Constants (must match preprocessing / model config)
# -----------------------------
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = 1024
N_MELS = 64
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_vae(device: torch.device) -> Tuple[AutoencoderKL, float]:
    """Load the same VAE used during preprocessing and return model + scaling."""
    vae = AutoencoderKL.from_pretrained("cvssp/audioldm-s-full-v2", subfolder="vae")
    vae = vae.to(device)
    vae.eval()
    scaling = vae.config.scaling_factor
    return vae, scaling


def load_hifigan(repo_id: str, subfolder: str, device: torch.device) -> SpeechT5HifiGan:
    """Load a HiFi-GAN vocoder. Defaults to AudioLDM’s own 16k / 64-mel model."""
    vocoder = SpeechT5HifiGan.from_pretrained(repo_id, subfolder=subfolder)
    vocoder = vocoder.to(device)
    vocoder.eval()
    return vocoder


def decode_latent(vae: AutoencoderKL, scaling: float, z: torch.Tensor) -> torch.Tensor:
    """Decode latent to log-mel spectrogram.

    Args:
        vae: loaded AutoencoderKL
        scaling: scaling_factor from VAE config
        z: latent tensor (B, C, H, W)
    Returns:
        log-mel tensor in model layout, shape (B, 1, H, W)
    """

    with torch.no_grad():
        log_mel = vae.decode(z / scaling).sample
    return log_mel


def decoded_log_mel_to_canonical(log_mel_4d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert decoded log-mel to canonical layouts.

    Returns:
        mel_bmt: (B, 64, T) for inversion/analysis
        mel_bt64: (B, T, 64) for SpeechT5HifiGan
    """

    mel_3d = log_mel_4d.squeeze(1)
    if mel_3d.dim() != 3:
        raise ValueError(f"Expected decoded mel to become 3D after squeeze, got {tuple(mel_3d.shape)}")

    # Support both latent conventions:
    # - old saved latents often decode to (B, 64, T)
    # - diffusers-native latents decode to (B, T, 64)
    if mel_3d.shape[1] == N_MELS:
        mel_bmt = mel_3d
    elif mel_3d.shape[2] == N_MELS:
        mel_bmt = mel_3d.transpose(1, 2)
    else:
        raise ValueError(f"Decoded mel has no 64-bin axis: {tuple(mel_3d.shape)}")

    mel_bt64 = mel_bmt.transpose(1, 2)
    return mel_bmt, mel_bt64


def mel_to_waveform(mel_power_bmt: torch.Tensor, device: torch.device, griffin_iters: int = 32) -> torch.Tensor:
    """Convert mel power spectrogram to waveform using Griffin-Lim.

    This is a fallback until a matched HiFi-GAN checkpoint (16k/64-mel) is wired in.
    """

    # Inverse Mel -> magnitude spectrogram
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=N_FFT // 2 + 1,
        n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
        f_min=0,
        f_max=8000,
        mel_scale="htk",
        norm="slaney",
    ).to(device)

    magnitude = inv_mel(mel_power_bmt)

    # Griffin-Lim phase reconstruction
    window = torch.hann_window(WIN_LENGTH, device=device)
    waveform = torchaudio.functional.griffinlim(
        magnitude,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        power=1.0,
        n_iter=griffin_iters,
        momentum=0.99,
        length=None,
        rand_init=True,
    )

    return waveform


def load_latent_file(path: str, device: torch.device) -> Tuple[torch.Tensor, int]:
    obj = torch.load(path, map_location=device)
    z = obj["z"].float()  # stored as half; convert back to float32
    genre_id = int(obj["genre_id"])
    return z, genre_id


def iter_latent_files(input_path: str) -> Iterable[str]:
    if os.path.isdir(input_path):
        yield from sorted(glob.glob(os.path.join(input_path, "*.pt")))
    else:
        yield input_path


def save_waveform(waveform: torch.Tensor, path: str):
    # Accept (T), (B, T) or (1, B, T); flatten batch->channels for saving.
    if waveform.dim() == 3 and waveform.size(0) == 1:
        waveform = waveform.squeeze(0)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() > 2:
        # collapse all but last dim into channels
        waveform = waveform.view(-1, waveform.size(-1))
    waveform = waveform.clamp(-1.0, 1.0).cpu()  # (channels, T)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, waveform, SAMPLE_RATE)


def main():
    parser = argparse.ArgumentParser(description="Decode VAE latents and vocode to audio.")
    parser.add_argument(
        "--latents",
        default=str(PROJECT_ROOT / "data/processed_latents"),
        help="Path to a single latent .pt file or a directory containing many",
    )
    parser.add_argument(
        "--outdir",
        default=str(PROJECT_ROOT / "recon_audio"),
        help="Directory to write reconstructed .wav files",
    )
    parser.add_argument("--max_files", type=int, default=4, help="Stop after this many files (for quick smoke test)")
    parser.add_argument("--griffin_iters", type=int, default=32, help="Griffin-Lim iterations")
    parser.add_argument("--use_hifigan", action="store_true", help="Use HiFi-GAN (AudioLDM vocoder) instead of Griffin-Lim")
    parser.add_argument("--vocoder_repo", default="cvssp/audioldm-s-full-v2", help="HF repo id for the vocoder")
    parser.add_argument("--vocoder_subfolder", default="vocoder", help="Subfolder inside the repo for vocoder weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vae, scaling = load_vae(device)
    vocoder = None
    if args.use_hifigan:
        vocoder = load_hifigan(args.vocoder_repo, args.vocoder_subfolder, device)

    # Load genre mapping for reference (optional)
    genre_map_path = os.path.join(os.path.dirname(args.latents), "genre_mapping.json") if os.path.isdir(args.latents) else os.path.join(os.path.dirname(os.path.dirname(args.latents)), "genre_mapping.json")
    if os.path.exists(genre_map_path):
        with open(genre_map_path) as f:
            genre_map = json.load(f)
        inv_genre_map = {v: k for k, v in genre_map.items()}
    else:
        genre_map = None
        inv_genre_map = None

    for idx, latent_path in enumerate(iter_latent_files(args.latents)):
        if idx >= args.max_files:
            break

        print(f"Decoding {latent_path}")
        z, genre_id = load_latent_file(latent_path, device)

        log_mel = decode_latent(vae, scaling, z)
        mel_bmt, mel_bt64 = decoded_log_mel_to_canonical(log_mel)

        if vocoder:
            with torch.no_grad():
                vocoder_out = vocoder(mel_bt64)
            waveform = getattr(vocoder_out, "waveform", vocoder_out)
            # SpeechT5HifiGan returns (B, T) or (T,)
            if waveform.dim() == 2 and waveform.size(0) == 1:
                waveform = waveform.squeeze(0)
        else:
            mel_power = torch.exp(mel_bmt)
            waveform = mel_to_waveform(mel_power, device=device, griffin_iters=args.griffin_iters)

        track_id = os.path.splitext(os.path.basename(latent_path))[0]
        genre_name = inv_genre_map.get(genre_id, str(genre_id)) if inv_genre_map else str(genre_id)
        out_path = os.path.join(args.outdir, f"{track_id}_genre-{genre_name}.wav")
        save_waveform(waveform, out_path)
        print(f"Saved {out_path}")

    print("Done. For better audio, replace Griffin-Lim with a 16 kHz / 64-mel HiFi-GAN vocoder once available.")


if __name__ == "__main__":
    main()
