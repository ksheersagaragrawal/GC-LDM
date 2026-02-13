"""Genre conditioning module used by the diffusion UNet cross-attention path."""

from typing import Optional

import torch
import torch.nn as nn


class GenreCondition(nn.Module):
    """Maps genre ids to cross-attention token sequences."""

    def __init__(
        self,
        num_genres: int,
        token_count: int = 8,
        embedding_dim: int = 512,
        cross_attention_dim: int = 512,
        null_genre_id: Optional[int] = None,
    ):
        super().__init__()
        if num_genres <= 0:
            raise ValueError(f"num_genres must be > 0, got {num_genres}")
        if token_count <= 0:
            raise ValueError(f"token_count must be > 0, got {token_count}")

        self.num_genres = int(num_genres)
        self.token_count = int(token_count)
        self.embedding_dim = int(embedding_dim)
        self.cross_attention_dim = int(cross_attention_dim)
        self.null_genre_id = self.num_genres if null_genre_id is None else int(null_genre_id)

        self.vocab_size = max(self.num_genres, self.null_genre_id) + 1
        self.genre_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.genre_projection = nn.Linear(
            self.embedding_dim,
            self.token_count * self.cross_attention_dim,
        )

    def unconditional_ids(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return null-genre ids for unconditional forward passes."""
        return torch.full(
            (batch_size,),
            self.null_genre_id,
            dtype=torch.long,
            device=device,
        )

    def forward(
        self,
        genre_ids: Optional[torch.Tensor],
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        drop_condition_mask: Optional[torch.Tensor] = None,
        force_unconditional: bool = False,
    ) -> torch.Tensor:
        """Convert genre ids into cross-attention tokens.

        Args:
            genre_ids: shape [B]. May be None when force_unconditional=True.
            batch_size: required when genre_ids is None.
            device: optional device override when genre_ids is None.
            drop_condition_mask: optional bool mask [B]; masked rows use null token.
            force_unconditional: if True, ignores genre_ids and uses null token for all.
        """
        if genre_ids is None:
            if batch_size is None:
                raise ValueError("batch_size is required when genre_ids is None")
            if device is None:
                device = self.genre_embedding.weight.device
            genre_ids = self.unconditional_ids(batch_size=batch_size, device=device)
        else:
            genre_ids = genre_ids.to(dtype=torch.long)

        if force_unconditional:
            genre_ids = self.unconditional_ids(
                batch_size=genre_ids.shape[0],
                device=genre_ids.device,
            )

        if drop_condition_mask is not None:
            mask = drop_condition_mask.to(device=genre_ids.device, dtype=torch.bool).view(-1)
            if mask.shape[0] != genre_ids.shape[0]:
                raise ValueError(
                    "drop_condition_mask must have same batch size as genre_ids. "
                    f"Got {mask.shape[0]} and {genre_ids.shape[0]}"
                )
            genre_ids = genre_ids.clone()
            genre_ids[mask] = self.null_genre_id

        min_id = int(genre_ids.min().item())
        max_id = int(genre_ids.max().item())
        if min_id < 0 or max_id >= self.vocab_size:
            raise ValueError(
                f"genre_ids out of range [0, {self.vocab_size - 1}]. "
                f"Got min={min_id}, max={max_id}"
            )

        embedding = self.genre_embedding(genre_ids)
        final_tokens = self.genre_projection(embedding)
        batch_count = genre_ids.shape[0]
        final_tokens = final_tokens.reshape(
            batch_count,
            self.token_count,
            self.cross_attention_dim,
        )
        return final_tokens
