"""Sample genre-conditioned latents from a trained checkpoint and export audio."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from diffusers import DDPMScheduler

from scripts.decode_and_vocode import (
    decode_latent,
    decoded_log_mel_to_canonical,
    load_hifigan,
    load_vae,
    mel_to_waveform,
    save_waveform,
)
from scripts.diffusion_runtime import (
    build_model,
    load_genre_mapping,
    now_run_name,
    sample_latents_with_cfg,
    set_seed,
    to_jsonable,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample latents/audio from genre-conditioned diffusion model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--genre_mapping", type=str, default="data/processed_latents/genre_mapping.json")
    parser.add_argument("--output_dir", type=str, default="runs/samples")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--genre_ids", type=str, default="0", help="Comma-separated genre ids, e.g. '0,3,7'")
    parser.add_argument("--num_samples_per_genre", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seed_stride", type=int, default=1000)

    parser.add_argument("--cfg_scale", type=float, default=3.5)
    parser.add_argument("--num_steps", type=int, default=50)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--export_audio", dest="export_audio", action="store_true")
    parser.add_argument("--no_export_audio", dest="export_audio", action="store_false")
    parser.set_defaults(export_audio=True)
    parser.add_argument("--use_hifigan", action="store_true", help="If false, Griffin-Lim fallback is used")
    parser.add_argument("--griffin_iters", type=int, default=64)
    parser.add_argument("--vocoder_repo", type=str, default="cvssp/audioldm-s-full-v2")
    parser.add_argument("--vocoder_subfolder", type=str, default="vocoder")
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_genre_ids(genre_ids: str) -> List[int]:
    values = [x.strip() for x in genre_ids.split(",") if x.strip()]
    if not values:
        raise ValueError("genre_ids is empty")
    return [int(x) for x in values]


def infer_latent_shape(model: torch.nn.Module) -> Tuple[int, int, int, int]:
    if hasattr(model, "config") and all(
        hasattr(model.config, key) for key in ["latent_channels", "latent_height", "latent_width"]
    ):
        return (1, int(model.config.latent_channels), int(model.config.latent_height), int(model.config.latent_width))

    if hasattr(model, "unet") and hasattr(model.unet, "config"):
        in_channels = int(model.unet.config.in_channels)
        sample_size = model.unet.config.sample_size
        if isinstance(sample_size, int):
            height, width = sample_size, sample_size
        else:
            height, width = int(sample_size[0]), int(sample_size[1])
        return (1, in_channels, height, width)

    return (1, 8, 256, 16)


def append_jsonl(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def main():
    args = parse_args()
    device = pick_device(args.device)
    set_seed(args.seed)

    checkpoint_path = (PROJECT_ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    genre_mapping_path = (
        (PROJECT_ROOT / args.genre_mapping).resolve()
        if not Path(args.genre_mapping).is_absolute()
        else Path(args.genre_mapping)
    )
    output_root = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    payload = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(
        genre_mapping_path=str(genre_mapping_path),
        device=device,
        model_config=payload.get("model_config"),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    noise_scheduler = (
        DDPMScheduler.from_config(payload["noise_scheduler_config"])
        if payload.get("noise_scheduler_config") is not None
        else DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    )

    genre_mapping = load_genre_mapping(str(genre_mapping_path))
    inverse_genre_mapping = {int(v): k for k, v in genre_mapping.items()}
    num_genres = len(genre_mapping)
    requested_genres = parse_genre_ids(args.genre_ids)
    for gid in requested_genres:
        if gid < 0 or gid >= num_genres:
            raise ValueError(f"genre_id {gid} out of valid range [0, {num_genres - 1}]")

    run_name = args.run_name or now_run_name("sample")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    sampling_config = {
        "checkpoint": str(checkpoint_path),
        "genre_mapping": str(genre_mapping_path),
        "run_name": run_name,
        "cfg_scale": args.cfg_scale,
        "num_steps": args.num_steps,
        "genre_ids": requested_genres,
        "num_samples_per_genre": args.num_samples_per_genre,
        "seed": args.seed,
        "seed_stride": args.seed_stride,
        "device": str(device),
        "export_audio": args.export_audio,
        "use_hifigan": args.use_hifigan,
    }
    (run_dir / "sampling_config.json").write_text(json.dumps(to_jsonable(sampling_config), indent=2))

    latent_shape = infer_latent_shape(model)

    vae = None
    scaling = None
    vocoder = None
    if args.export_audio:
        vae, scaling = load_vae(device)
        if args.use_hifigan:
            vocoder = load_hifigan(args.vocoder_repo, args.vocoder_subfolder, device)

    metadata_path = run_dir / "samples_metadata.jsonl"

    for genre_index, genre_id in enumerate(requested_genres):
        genre_name = inverse_genre_mapping.get(int(genre_id), f"id{genre_id}")
        genre_dir = run_dir / f"genre_{genre_name}_{genre_id}"
        genre_dir.mkdir(parents=True, exist_ok=True)

        for sample_idx in range(args.num_samples_per_genre):
            seed_value = args.seed + genre_index * args.seed_stride + sample_idx
            latents = sample_latents_with_cfg(
                model=model,
                noise_scheduler=noise_scheduler,
                genre_id=genre_id,
                num_genres=num_genres,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_steps,
                seed=seed_value,
                latent_shape=latent_shape,
                device=device,
            )

            stem = f"sample_{sample_idx:03d}_seed_{seed_value}"
            latent_out = genre_dir / f"{stem}.pt"
            torch.save(
                {
                    "z": latents.detach().cpu().half(),
                    "genre_id": int(genre_id),
                    "seed": int(seed_value),
                    "cfg_scale": float(args.cfg_scale),
                    "num_steps": int(args.num_steps),
                },
                latent_out,
            )

            audio_path = None
            if args.export_audio:
                log_mel = decode_latent(vae, scaling, latents)
                mel_bmt, mel_bt64 = decoded_log_mel_to_canonical(log_mel)
                if vocoder is not None:
                    with torch.no_grad():
                        vocoder_out = vocoder(mel_bt64)
                    waveform = getattr(vocoder_out, "waveform", vocoder_out)
                else:
                    mel_power = torch.exp(mel_bmt)
                    waveform = mel_to_waveform(mel_power, device=device, griffin_iters=args.griffin_iters)

                audio_path = genre_dir / f"{stem}.wav"
                save_waveform(waveform, str(audio_path))

            metadata_row = {
                "genre_id": int(genre_id),
                "genre_name": genre_name,
                "seed": int(seed_value),
                "cfg_scale": float(args.cfg_scale),
                "num_steps": int(args.num_steps),
                "latent_path": str(latent_out),
                "audio_path": str(audio_path) if audio_path is not None else None,
            }
            append_jsonl(metadata_path, metadata_row)
            print(f"[sample] genre={genre_name} seed={seed_value} latent={latent_out}")

    print(f"Sampling complete. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
