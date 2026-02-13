"""Runtime helpers shared by training, sampling, and evaluation scripts."""

import inspect
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from diffusers import DDPMScheduler

from models.genre_diffusion_model import GenreConditionedUNet


def now_run_name(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_genre_mapping(genre_mapping_path: str) -> Dict[str, int]:
    with open(genre_mapping_path, "r") as f:
        return json.load(f)


def get_null_genre_id(model: torch.nn.Module, default_num_genres: int) -> int:
    if hasattr(model, "genre_condition") and hasattr(model.genre_condition, "null_genre_id"):
        return int(model.genre_condition.null_genre_id)
    return int(default_num_genres)


def build_model(
    genre_mapping_path: str,
    device: torch.device,
    model_config: Optional[Dict] = None,
) -> torch.nn.Module:
    """Build model for both old and new GenreConditionedUNet APIs."""
    constructor = inspect.signature(GenreConditionedUNet.__init__)

    if model_config is not None and "config" in constructor.parameters:
        # New API: GenreConditionedUNet(config=...)
        from models.genre_diffusion_model import GenreDiffusionConfig  # imported lazily for compatibility

        cfg = (
            GenreDiffusionConfig.from_dict(model_config)
            if hasattr(GenreDiffusionConfig, "from_dict")
            else GenreDiffusionConfig(**model_config)
        )
        model = GenreConditionedUNet(config=cfg)
    elif hasattr(GenreConditionedUNet, "from_genre_mapping"):
        model = GenreConditionedUNet.from_genre_mapping(genre_mapping_path)
    else:
        # Old API: no config args.
        model = GenreConditionedUNet()

    model = model.to(device)
    return model


def model_predict_noise(
    model: torch.nn.Module,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    genre_ids: torch.Tensor,
    *,
    num_genres: int,
    drop_condition_mask: Optional[torch.Tensor] = None,
    force_unconditional: bool = False,
) -> torch.Tensor:
    """Forward wrapper compatible with both old/new model signatures."""
    try:
        return model(
            noisy_latents,
            timesteps,
            genre_ids=genre_ids,
            drop_condition_mask=drop_condition_mask,
            force_unconditional=force_unconditional,
        )
    except TypeError:
        # Old API fallback: model(noisy_latents, timesteps, genre_ids)
        null_id = get_null_genre_id(model=model, default_num_genres=num_genres)
        ids = genre_ids.clone()
        if force_unconditional:
            ids[:] = null_id
        if drop_condition_mask is not None:
            mask = drop_condition_mask.to(device=ids.device, dtype=torch.bool).view(-1)
            ids[mask] = null_id
        return model(noisy_latents, timesteps, ids)


def create_noise_scheduler(num_train_timesteps: int = 1000) -> DDPMScheduler:
    return DDPMScheduler(num_train_timesteps=num_train_timesteps, prediction_type="epsilon")


def save_training_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    noise_scheduler: DDPMScheduler,
    run_config: Dict[str, Any],
    *,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    seed: int,
):
    payload = {
        "checkpoint_type": "genre_diffusion_train_state",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "noise_scheduler_config": dict(noise_scheduler.config),
        "run_config": to_jsonable(run_config),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "seed": int(seed),
        "model_config": (
            model.config.to_dict() if hasattr(model, "config") and hasattr(model.config, "to_dict") else None
        ),
        "genre_mapping": getattr(model, "genre_mapping", None),
        # Keep explicit module states to satisfy handoff requirements.
        "unet_state_dict": model.unet.state_dict() if hasattr(model, "unet") else None,
        "genre_condition_state_dict": (
            model.genre_condition.state_dict() if hasattr(model, "genre_condition") else None
        ),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_training_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    *,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if lr_scheduler is not None and payload.get("lr_scheduler_state_dict") is not None:
        lr_scheduler.load_state_dict(payload["lr_scheduler_state_dict"])

    # Restore RNG state for deterministic resume when available.
    if "torch_rng_state" in payload and payload["torch_rng_state"] is not None:
        torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        try:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        except Exception:
            pass
    return payload


@torch.no_grad()
def sample_latents_with_cfg(
    model: torch.nn.Module,
    noise_scheduler: DDPMScheduler,
    *,
    genre_id: int,
    num_genres: int,
    cfg_scale: float,
    num_inference_steps: int,
    seed: int,
    latent_shape: tuple,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(latent_shape, generator=generator, device=device)
    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    genre_ids = torch.tensor([genre_id] * latent_shape[0], dtype=torch.long, device=device)

    for timestep in noise_scheduler.timesteps:
        timestep_batch = torch.full(
            (latent_shape[0],),
            int(timestep.item()) if torch.is_tensor(timestep) else int(timestep),
            dtype=torch.long,
            device=device,
        )
        eps_cond = model_predict_noise(
            model=model,
            noisy_latents=latents,
            timesteps=timestep_batch,
            genre_ids=genre_ids,
            num_genres=num_genres,
        )
        if cfg_scale != 1.0:
            eps_uncond = model_predict_noise(
                model=model,
                noisy_latents=latents,
                timesteps=timestep_batch,
                genre_ids=genre_ids,
                num_genres=num_genres,
                force_unconditional=True,
            )
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_cond

        latents = noise_scheduler.step(eps, timestep, latents).prev_sample

    return latents
