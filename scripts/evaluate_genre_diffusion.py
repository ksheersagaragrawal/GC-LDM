"""Evaluate trained genre-conditioned diffusion checkpoints."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from tqdm import tqdm

from scripts.diffusion_runtime import (
    build_model,
    load_genre_mapping,
    model_predict_noise,
    now_run_name,
    set_seed,
    to_jsonable,
)
from scripts.latent_dataset import get_dataloader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate genre diffusion model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="data/processed_latents/manifest.csv")
    parser.add_argument("--genre_mapping", type=str, default="data/processed_latents/genre_mapping.json")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--output_dir", type=str, default="runs/eval")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument(
        "--generated_metadata",
        type=str,
        default=None,
        help="Path to samples_metadata.jsonl from sample script for genre consistency proxy",
    )
    parser.add_argument("--train_split_for_centroids", type=str, default="train")
    parser.add_argument("--max_centroid_batches", type=int, default=300)

    parser.add_argument("--compute_fad", action="store_true")
    parser.add_argument("--reference_audio_dir", type=str, default=None)
    parser.add_argument("--generated_audio_dir", type=str, default=None)
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_val_loss(
    model: torch.nn.Module,
    val_loader,
    noise_scheduler,
    *,
    device: torch.device,
    num_genres: int,
    max_batches: int,
) -> Tuple[float, int]:
    model.eval()
    losses = []
    batches = 0
    for batch in val_loader:
        latents, genre_ids = batch[:2]
        latents = latents.to(device=device, dtype=torch.float32)
        genre_ids = genre_ids.to(device=device, dtype=torch.long)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device,
            dtype=torch.long,
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        pred_noise = model_predict_noise(
            model=model,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            genre_ids=genre_ids,
            num_genres=num_genres,
        )
        losses.append(float(F.mse_loss(pred_noise, noise).item()))
        batches += 1
        if max_batches > 0 and batches >= max_batches:
            break
    return sum(losses) / max(1, len(losses)), batches


@torch.no_grad()
def compute_genre_centroids(
    loader,
    *,
    max_batches: int,
) -> Dict[int, torch.Tensor]:
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = defaultdict(int)
    batches = 0

    for batch in loader:
        latents, genre_ids = batch[:2]
        latents = latents.float()
        genre_ids = genre_ids.long()
        flat = latents.view(latents.shape[0], -1)
        for idx in range(flat.shape[0]):
            gid = int(genre_ids[idx].item())
            if gid not in sums:
                sums[gid] = flat[idx].clone()
            else:
                sums[gid] += flat[idx]
            counts[gid] += 1
        batches += 1
        if max_batches > 0 and batches >= max_batches:
            break

    centroids = {gid: sums[gid] / counts[gid] for gid in sums}
    return centroids


def load_generated_rows(metadata_path: Path) -> List[Dict]:
    rows = []
    with open(metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def nearest_centroid(centroids: Dict[int, torch.Tensor], vector: torch.Tensor) -> int:
    best_gid = None
    best_dist = None
    for gid, centroid in centroids.items():
        dist = torch.norm(vector - centroid, p=2).item()
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_gid = gid
    return int(best_gid)


def compute_genre_consistency_proxy(
    centroids: Dict[int, torch.Tensor],
    metadata_rows: List[Dict],
) -> Dict[str, float]:
    total = 0
    correct = 0
    for row in metadata_rows:
        latent_path = row.get("latent_path")
        target_gid = int(row.get("genre_id"))
        if latent_path is None or not Path(latent_path).exists():
            continue
        obj = torch.load(latent_path, map_location="cpu")
        z = obj["z"].float()
        if z.ndim == 4 and z.shape[0] == 1:
            z = z.squeeze(0)
        vector = z.reshape(-1)
        pred_gid = nearest_centroid(centroids, vector)
        total += 1
        if pred_gid == target_gid:
            correct += 1
    acc = (correct / total) if total > 0 else 0.0
    return {
        "genre_consistency_proxy_acc": acc,
        "genre_consistency_proxy_total": total,
    }


def maybe_compute_fad(reference_audio_dir: str, generated_audio_dir: str) -> Dict[str, float]:
    try:
        from frechet_audio_distance import FrechetAudioDistance
    except Exception as exc:
        return {"fad_error": f"frechet_audio_distance not available: {exc}"}

    try:
        fad = FrechetAudioDistance(
            model_name="vggish",
            sample_rate=16000,
            use_pca=False,
            use_activation=False,
            verbose=False,
        )
        score = fad.score(reference_audio_dir, generated_audio_dir)
        return {"fad": float(score)}
    except Exception as exc:
        return {"fad_error": str(exc)}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    checkpoint_path = (PROJECT_ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    manifest_path = (PROJECT_ROOT / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    genre_mapping_path = (
        (PROJECT_ROOT / args.genre_mapping).resolve()
        if not Path(args.genre_mapping).is_absolute()
        else Path(args.genre_mapping)
    )
    output_root = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    run_name = args.run_name or now_run_name("eval")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(
        genre_mapping_path=str(genre_mapping_path),
        device=device,
        model_config=payload.get("model_config"),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    genre_mapping = load_genre_mapping(str(genre_mapping_path))
    num_genres = len(genre_mapping)
    noise_scheduler = (
        DDPMScheduler.from_config(payload["noise_scheduler_config"])
        if payload.get("noise_scheduler_config") is not None
        else DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")
    )

    val_loader = get_dataloader(
        manifest_path=str(manifest_path),
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        dtype=torch.float32,
    )

    results = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "genre_mapping": str(genre_mapping_path),
        "split": args.split,
        "seed": args.seed,
        "device": str(device),
    }

    val_loss, val_batches = compute_val_loss(
        model=model,
        val_loader=val_loader,
        noise_scheduler=noise_scheduler,
        device=device,
        num_genres=num_genres,
        max_batches=args.max_val_batches,
    )
    results["val_noise_mse"] = float(val_loss)
    results["val_batches"] = int(val_batches)

    if args.generated_metadata:
        generated_metadata_path = (
            (PROJECT_ROOT / args.generated_metadata).resolve()
            if not Path(args.generated_metadata).is_absolute()
            else Path(args.generated_metadata)
        )
        train_loader_for_centroids = get_dataloader(
            manifest_path=str(manifest_path),
            split=args.train_split_for_centroids,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            dtype=torch.float32,
        )
        centroids = compute_genre_centroids(
            train_loader_for_centroids,
            max_batches=args.max_centroid_batches,
        )
        metadata_rows = load_generated_rows(generated_metadata_path)
        results.update(compute_genre_consistency_proxy(centroids, metadata_rows))
        results["generated_metadata"] = str(generated_metadata_path)

    if args.compute_fad:
        if not args.reference_audio_dir or not args.generated_audio_dir:
            results["fad_error"] = "reference_audio_dir and generated_audio_dir are required when --compute_fad is set"
        else:
            results.update(maybe_compute_fad(args.reference_audio_dir, args.generated_audio_dir))

    (run_dir / "eval_results.json").write_text(json.dumps(to_jsonable(results), indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved eval results to {run_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
