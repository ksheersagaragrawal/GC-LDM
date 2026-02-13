"""Sanity test for AudioLDM VAE + HiFi-GAN reconstruction on a single clip.

This follows the input convention expected by `cvssp/audioldm-s-full-v2`:
  1) waveform -> log-mel via TacotronSTFT
  2) feed raw log-mel (not mean/std-normalized) to VAE as (B, 1, T, F)
  3) feed log-mel to HiFi-GAN as (B, T, 64)
"""

from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from diffusers import AutoencoderKL
from librosa.filters import mel as librosa_mel_fn
from librosa.util import pad_center
from scipy.signal import get_window
from transformers import SpeechT5HifiGan

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
NUM_SAMPLES = 160000
NUM_FRAMES = 1024
INPUT_AUDIO = PROJECT_ROOT / "data/100/100478.mp3"
OUTPUT_DIR = PROJECT_ROOT / "sanity_test_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class STFT(torch.nn.Module):
    def __init__(self, filter_length, hop_length, win_length, window="hann"):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        scale = filter_length / hop_length
        fourier_basis = np.fft.fft(np.eye(filter_length))
        cutoff = int(filter_length / 2 + 1)
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        fft_window = get_window(window, win_length, fftbins=True)
        fft_window = pad_center(fft_window, size=filter_length)
        fft_window = torch.from_numpy(fft_window).float()
        forward_basis *= fft_window
        inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = torch.nn.functional.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)
        forward_transform = torch.nn.functional.conv1d(
            input_data, self.forward_basis, stride=self.hop_length, padding=0
        )
        cutoff = int(self.filter_length / 2 + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        return magnitude


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length, hop_length, win_length, n_mel_channels, sampling_rate, mel_fmin, mel_fmax):
        super().__init__()
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

    def mel_spectrogram(self, y):
        magnitudes = self.stft_fn.transform(y)
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = torch.log(torch.clamp(mel_output, min=1e-5))
        return mel_output


def pad_or_trim_to_10s(audio: np.ndarray) -> np.ndarray:
    audio = audio[:NUM_SAMPLES]
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
    return audio


def pad_or_trim_mel_frames(log_mel: torch.Tensor) -> torch.Tensor:
    if log_mel.shape[2] >= NUM_FRAMES:
        return log_mel[:, :, :NUM_FRAMES]
    return torch.nn.functional.pad(log_mel, (0, NUM_FRAMES - log_mel.shape[2]))


def align_and_metrics(reference: torch.Tensor, estimate: torch.Tensor):
    target_len = min(reference.shape[-1], estimate.shape[-1])
    ref = reference[:, :target_len]
    est = estimate[:, :target_len]
    mse = ((ref - est) ** 2).mean().item()
    mae = (ref - est).abs().mean().item()
    cos = torch.nn.functional.cosine_similarity(ref, est, dim=-1).mean().item()
    return mse, mae, cos


def main():
    print("=" * 60)
    print("Loading audio")
    print("=" * 60)
    audio, _ = librosa.load(str(INPUT_AUDIO), sr=SAMPLE_RATE, mono=True)
    audio = pad_or_trim_to_10s(audio)
    audio = audio - np.mean(audio)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    audio = 0.5 * audio
    waveform = torch.FloatTensor(audio).unsqueeze(0)
    torchaudio.save(str(OUTPUT_DIR / "original.wav"), waveform, SAMPLE_RATE)
    print(f"  Saved {OUTPUT_DIR / 'original.wav'}")

    print("\n" + "=" * 60)
    print("Computing log-mel")
    print("=" * 60)
    stft_fn = TacotronSTFT(
        filter_length=1024,
        hop_length=160,
        win_length=1024,
        n_mel_channels=64,
        sampling_rate=16000,
        mel_fmin=0,
        mel_fmax=8000,
    )
    log_mel = stft_fn.mel_spectrogram(waveform)  # (1, 64, T)
    log_mel = pad_or_trim_mel_frames(log_mel)    # (1, 64, 1024)
    print(f"  log_mel shape: {tuple(log_mel.shape)}")
    print(f"  log_mel range: [{log_mel.min():.3f}, {log_mel.max():.3f}]")
    print(f"  log_mel mean/std: {log_mel.mean():.3f} / {log_mel.std():.3f}")

    print("\n" + "=" * 60)
    print("VAE round-trip")
    print("=" * 60)
    vae = AutoencoderKL.from_pretrained("cvssp/audioldm-s-full-v2", subfolder="vae")
    vae.eval()
    scaling = vae.config.scaling_factor

    # VAE expects (B, 1, T, F) = (B, 1, 1024, 64).
    vae_input = log_mel.transpose(1, 2).unsqueeze(1)
    with torch.no_grad():
        z = vae.encode(vae_input).latent_dist.mode() * scaling
        vae_recon = vae.decode(z / scaling).sample
    vae_mse = ((vae_input - vae_recon) ** 2).mean().item()
    vae_mae = (vae_input - vae_recon).abs().mean().item()
    print(f"  input shape:  {tuple(vae_input.shape)}")
    print(f"  latent shape: {tuple(z.shape)}")
    print(f"  recon shape:  {tuple(vae_recon.shape)}")
    print(f"  VAE log-mel MSE: {vae_mse:.4f}")
    print(f"  VAE log-mel MAE: {vae_mae:.4f}")

    print("\n" + "=" * 60)
    print("Vocoding")
    print("=" * 60)
    vocoder = SpeechT5HifiGan.from_pretrained("cvssp/audioldm-s-full-v2", subfolder="vocoder")
    vocoder.eval()

    with torch.no_grad():
        # Direct vocoder baseline (no VAE), useful to isolate mel/vocoder mismatch.
        wav_direct = vocoder(log_mel.transpose(1, 2)).clamp(-1, 1)
        wav_recon = vocoder(vae_recon.squeeze(1)).clamp(-1, 1)

    if wav_direct.dim() == 1:
        wav_direct = wav_direct.unsqueeze(0)
    if wav_recon.dim() == 1:
        wav_recon = wav_recon.unsqueeze(0)

    torchaudio.save(str(OUTPUT_DIR / "direct_vocoder.wav"), wav_direct.cpu(), SAMPLE_RATE)
    torchaudio.save(str(OUTPUT_DIR / "reconstructed.wav"), wav_recon.cpu(), SAMPLE_RATE)
    print(f"  Saved {OUTPUT_DIR / 'direct_vocoder.wav'}")
    print(f"  Saved {OUTPUT_DIR / 'reconstructed.wav'}")

    direct_mse, direct_mae, direct_cos = align_and_metrics(waveform, wav_direct)
    recon_mse, recon_mae, recon_cos = align_and_metrics(waveform, wav_recon)
    print("\nWaveform similarity vs original (lower MSE/MAE is better):")
    print(f"  direct vocoder: MSE={direct_mse:.4f}, MAE={direct_mae:.4f}, COS={direct_cos:.4f}")
    print(f"  vae+vocoder   : MSE={recon_mse:.4f}, MAE={recon_mae:.4f}, COS={recon_cos:.4f}")

    print("\nCompare these files:")
    print(f"  {OUTPUT_DIR}/original.wav")
    print(f"  {OUTPUT_DIR}/direct_vocoder.wav")
    print(f"  {OUTPUT_DIR}/reconstructed.wav")


if __name__ == "__main__":
    main()
