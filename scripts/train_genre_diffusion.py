"""Train genre-conditioned latent diffusion model."""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from diffusers import DDPMScheduler
from scripts.diffusion_runtime import (
    build_model,
    create_noise_scheduler,
    load_genre_mapping,
    load_training_checkpoint,
    model_predict_noise,
    now_run_name,
    save_training_checkpoint,
    set_seed,
    to_jsonable,
)
from scripts.latent_dataset import get_dataloader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train genre-conditioned latent diffusion model.")
    parser.add_argument("--manifest", type=str, default="data/processed_latents/manifest.csv")
    parser.add_argument("--genre_mapping", type=str, default="data/processed_latents/genre_mapping.json")
    parser.add_argument("--output_dir", type=str, default="runs/train_genre_diffusion")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=0, help="0 means use full epochs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.1)

    parser.add_argument("--val_every_steps", type=int, default=500)
    parser.add_argument("--save_every_steps", type=int, default=500)
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--max_val_batches", type=int, default=100)

    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def append_jsonl(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    val_loader,
    noise_scheduler,
    *,
    device: torch.device,
    num_genres: int,
    max_val_batches: int,
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
        loss = F.mse_loss(pred_noise, noise)
        losses.append(float(loss.item()))
        batches += 1
        if max_val_batches > 0 and batches >= max_val_batches:
            break
    mean_loss = sum(losses) / max(1, len(losses))
    return mean_loss, batches


def main():
    args = parse_args()
    device = pick_device(args.device)
    set_seed(args.seed)

    manifest_path = (PROJECT_ROOT / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    genre_mapping_path = (
        (PROJECT_ROOT / args.genre_mapping).resolve()
        if not Path(args.genre_mapping).is_absolute()
        else Path(args.genre_mapping)
    )

    run_name = args.run_name or now_run_name("train")
    run_dir = ((PROJECT_ROOT / args.output_dir) / run_name).resolve() if not Path(args.output_dir).is_absolute() else (Path(args.output_dir) / run_name).resolve()
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader = get_dataloader(
        manifest_path=str(manifest_path),
        split=args.train_split,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        dtype=torch.float32,
    )
    val_loader = get_dataloader(
        manifest_path=str(manifest_path),
        split=args.val_split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        dtype=torch.float32,
    )
    genre_mapping = load_genre_mapping(str(genre_mapping_path))
    num_genres = len(genre_mapping)

    resume_payload = None
    model_config = None
    if args.resume_checkpoint:
        resume_payload = torch.load(args.resume_checkpoint, map_location="cpu")
        model_config = resume_payload.get("model_config")

    model = build_model(
        genre_mapping_path=str(genre_mapping_path),
        device=device,
        model_config=model_config,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    noise_scheduler = (
        create_noise_scheduler(args.num_train_timesteps)
        if resume_payload is None or resume_payload.get("noise_scheduler_config") is None
        else DDPMScheduler.from_config(resume_payload["noise_scheduler_config"])
    )

    if args.max_train_steps > 0:
        total_steps = args.max_train_steps
    else:
        total_steps = args.epochs * len(train_loader)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    use_amp = args.use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume_checkpoint:
        payload = load_training_checkpoint(
            checkpoint_path=args.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            map_location="cpu",
            strict=True,
        )
        global_step = int(payload.get("global_step", 0))
        start_epoch = int(payload.get("epoch", 0))
        best_val_loss = float(payload.get("best_val_loss", float("inf")))
        print(f"Resumed from {args.resume_checkpoint} at epoch={start_epoch}, global_step={global_step}")

    run_config = {
        "manifest": str(manifest_path),
        "genre_mapping": str(genre_mapping_path),
        "output_dir": str(run_dir),
        "run_name": run_name,
        "device": str(device),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_train_steps": args.max_train_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "num_train_timesteps": args.num_train_timesteps,
        "cfg_dropout_prob": args.cfg_dropout_prob,
        "val_every_steps": args.val_every_steps,
        "save_every_steps": args.save_every_steps,
        "max_val_batches": args.max_val_batches,
        "seed": args.seed,
    }
    (run_dir / "run_config.json").write_text(json.dumps(to_jsonable(run_config), indent=2))

    train_log_path = run_dir / "train_log.jsonl"
    val_log_path = run_dir / "val_log.jsonl"

    model.train()
    pbar = tqdm(total=total_steps, initial=min(global_step, total_steps), desc="Training")
    stop_training = False

    for epoch in range(start_epoch, args.epochs):
        for batch in train_loader:
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
            drop_mask = torch.rand(latents.shape[0], device=device) < args.cfg_dropout_prob

            with autocast(enabled=use_amp):
                pred_noise = model_predict_noise(
                    model=model,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    genre_ids=genre_ids,
                    num_genres=num_genres,
                    drop_condition_mask=drop_mask,
                )
                loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            global_step += 1
            pbar.update(1)

            if global_step % args.log_every_steps == 0:
                log_row = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                append_jsonl(train_log_path, log_row)

            if args.val_every_steps > 0 and global_step % args.val_every_steps == 0:
                val_loss, val_batches = run_validation(
                    model=model,
                    val_loader=val_loader,
                    noise_scheduler=noise_scheduler,
                    device=device,
                    num_genres=num_genres,
                    max_val_batches=args.max_val_batches,
                )
                model.train()
                val_row = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "val_loss": float(val_loss),
                    "val_batches": val_batches,
                }
                append_jsonl(val_log_path, val_row)
                print(f"[val] step={global_step} loss={val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_training_checkpoint(
                        checkpoint_path=str(ckpt_dir / "best.pt"),
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        noise_scheduler=noise_scheduler,
                        run_config=run_config,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        seed=args.seed,
                    )

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save_training_checkpoint(
                    checkpoint_path=str(ckpt_dir / "latest.pt"),
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    noise_scheduler=noise_scheduler,
                    run_config=run_config,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    seed=args.seed,
                )

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                stop_training = True
                break
        if stop_training:
            break

    # Final save
    save_training_checkpoint(
        checkpoint_path=str(ckpt_dir / "latest.pt"),
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        noise_scheduler=noise_scheduler,
        run_config=run_config,
        epoch=epoch if "epoch" in locals() else 0,
        global_step=global_step,
        best_val_loss=best_val_loss,
        seed=args.seed,
    )
    pbar.close()
    print(f"Training complete. latest checkpoint: {ckpt_dir / 'latest.pt'}")
    print(f"Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
