"""Genre-conditioned latent diffusion model wrapper."""

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

from models.gen_conditioning import GenreCondition


@dataclass
class GenreDiffusionConfig:
    """Configuration contract shared between train and sample code."""

    num_genres: int
    token_count: int = 8
    embedding_dim: int = 512
    cross_attention_dim: int = 512
    null_genre_id: Optional[int] = None

    latent_height: int = 256
    latent_width: int = 16
    latent_channels: int = 8

    layers_per_block: int = 2
    block_out_channels: Tuple[int, int, int, int] = (128, 256, 512, 1024)
    down_block_types: Tuple[str, str, str, str] = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    )
    up_block_types: Tuple[str, str, str, str] = (
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    )
    attention_head_dim: int = 8

    def __post_init__(self):
        if self.num_genres <= 0:
            raise ValueError(f"num_genres must be > 0, got {self.num_genres}")
        if self.null_genre_id is None:
            self.null_genre_id = self.num_genres

    @classmethod
    def from_genre_mapping(
        cls,
        genre_mapping_path: str,
        **kwargs,
    ) -> "GenreDiffusionConfig":
        with open(genre_mapping_path, "r") as f:
            mapping = json.load(f)
        return cls(num_genres=len(mapping), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict) -> "GenreDiffusionConfig":
        cooked = dict(data)
        # Convert serialized lists back to tuples where needed.
        if "block_out_channels" in cooked:
            cooked["block_out_channels"] = tuple(cooked["block_out_channels"])
        if "down_block_types" in cooked:
            cooked["down_block_types"] = tuple(cooked["down_block_types"])
        if "up_block_types" in cooked:
            cooked["up_block_types"] = tuple(cooked["up_block_types"])
        return cls(**cooked)

    def to_dict(self) -> Dict:
        return asdict(self)


class GenreConditionedUNet(nn.Module):
    """Diffusion UNet with genre-token cross-attention conditioning."""

    def __init__(
        self,
        config: Optional[GenreDiffusionConfig] = None,
        genre_mapping: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        if config is None:
            inferred_genres = len(genre_mapping) if genre_mapping is not None else 8
            config = GenreDiffusionConfig(num_genres=inferred_genres)
        if genre_mapping is not None and len(genre_mapping) != config.num_genres:
            raise ValueError(
                f"genre_mapping size ({len(genre_mapping)}) does not match "
                f"config.num_genres ({config.num_genres})"
            )
        self.config = config
        self.genre_mapping = genre_mapping

        self.unet = UNet2DConditionModel(
            sample_size=(config.latent_height, config.latent_width),
            in_channels=config.latent_channels,
            out_channels=config.latent_channels,
            layers_per_block=config.layers_per_block,
            block_out_channels=config.block_out_channels,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
            cross_attention_dim=config.cross_attention_dim,
            attention_head_dim=config.attention_head_dim,
        )
        self.genre_condition = GenreCondition(
            num_genres=config.num_genres,
            token_count=config.token_count,
            embedding_dim=config.embedding_dim,
            cross_attention_dim=config.cross_attention_dim,
            null_genre_id=config.null_genre_id,
        )

    @classmethod
    def from_genre_mapping(
        cls,
        genre_mapping_path: str,
        **kwargs,
    ) -> "GenreConditionedUNet":
        with open(genre_mapping_path, "r") as f:
            mapping = json.load(f)
        config = GenreDiffusionConfig(num_genres=len(mapping), **kwargs)
        return cls(config=config, genre_mapping=mapping)

    def _normalize_timesteps(
        self,
        timesteps: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, dtype=torch.long, device=device)
        else:
            timesteps = timesteps.to(device=device)

        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        if timesteps.shape[0] == 1 and batch_size > 1:
            timesteps = timesteps.expand(batch_size)
        if timesteps.shape[0] != batch_size:
            raise ValueError(
                "timesteps batch mismatch. "
                f"Expected {batch_size}, got {timesteps.shape[0]}"
            )
        return timesteps.long()

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        genre_ids: Optional[torch.Tensor] = None,
        drop_condition_mask: Optional[torch.Tensor] = None,
        force_unconditional: bool = False,
    ) -> torch.Tensor:
        """Forward pass for epsilon/noise prediction.

        Args:
            noisy_latents: latent tensor [B, C, H, W], expected C=8, H=256, W=16.
            timesteps: scalar or [B] diffusion step indices.
            genre_ids: [B] long tensor with genre ids.
            drop_condition_mask: optional bool mask [B] for CFG condition dropout.
            force_unconditional: when True, uses null condition for every sample.
        Returns:
            Predicted noise tensor with same shape as noisy_latents.
        """
        if noisy_latents.ndim != 4:
            raise ValueError(f"noisy_latents must be 4D [B,C,H,W], got {tuple(noisy_latents.shape)}")
        if noisy_latents.shape[1] != self.config.latent_channels:
            raise ValueError(
                f"expected latent channels={self.config.latent_channels}, "
                f"got {noisy_latents.shape[1]}"
            )
        if noisy_latents.shape[2] != self.config.latent_height or noisy_latents.shape[3] != self.config.latent_width:
            raise ValueError(
                f"expected latent spatial size=({self.config.latent_height}, {self.config.latent_width}), "
                f"got ({noisy_latents.shape[2]}, {noisy_latents.shape[3]})"
            )

        batch_size = noisy_latents.shape[0]
        device = noisy_latents.device
        timesteps = self._normalize_timesteps(timesteps=timesteps, batch_size=batch_size, device=device)
        genre_tokens = self.genre_condition(
            genre_ids=genre_ids,
            batch_size=batch_size,
            device=device,
            drop_condition_mask=drop_condition_mask,
            force_unconditional=force_unconditional,
        )
        out = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=genre_tokens,
        )
        return out.sample

    def save_model(self, checkpoint_path: str):
        """Save model weights and config together."""
        directory = os.path.dirname(checkpoint_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "model_config": self.config.to_dict(),
            "genre_mapping": self.genre_mapping,
            "checkpoint_type": "genre_conditioned_unet",
        }
        torch.save(payload, checkpoint_path)

    @classmethod
    def load_model(
        cls,
        checkpoint_path: str,
        map_location: Optional[str] = "cpu",
        strict: bool = True,
    ) -> "GenreConditionedUNet":
        """Load a model checkpoint produced by save_model()."""
        payload = torch.load(checkpoint_path, map_location=map_location)
        if "model_config" not in payload or "state_dict" not in payload:
            raise ValueError("Checkpoint missing required keys: model_config and state_dict")
        config = GenreDiffusionConfig.from_dict(payload["model_config"])
        model = cls(config=config, genre_mapping=payload.get("genre_mapping"))
        model.load_state_dict(payload["state_dict"], strict=strict)
        return model
