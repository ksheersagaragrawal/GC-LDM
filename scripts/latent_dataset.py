"""Latent dataset and dataloader utilities for diffusion training."""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LatentDataset(Dataset):
    """Load latent tensors and genre ids using a manifest file."""

    def __init__(
        self,
        manifest_path: str,
        split: str,
        dtype: torch.dtype = torch.float32,
        return_metadata: bool = False,
        root_dir: Optional[str] = None,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.dtype = dtype
        self.return_metadata = return_metadata
        self.root_dir = Path(root_dir).resolve() if root_dir else PROJECT_ROOT
        self.rows = self._load_rows()

    def _load_rows(self) -> List[Dict]:
        rows: List[Dict] = []
        with open(self.manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"latent_path", "track_id", "slice_id", "genre_id", "split"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"manifest missing required columns {required}, got {reader.fieldnames}"
                )
            for row in reader:
                if row["split"] != self.split:
                    continue
                rows.append(row)
        if not rows:
            raise ValueError(
                f"No rows found for split='{self.split}' in manifest: {self.manifest_path}"
            )
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, latent_path: str) -> Path:
        p = Path(latent_path)
        if p.is_absolute():
            return p
        if str(p).startswith("."):
            return (self.manifest_path.parent / p).resolve()
        return (self.root_dir / p).resolve()

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        latent_path = self._resolve_path(row["latent_path"])
        obj = torch.load(latent_path, map_location="cpu")
        if "z" not in obj or "genre_id" not in obj:
            raise KeyError(f"Missing 'z' or 'genre_id' in latent file: {latent_path}")

        z = obj["z"]
        if not torch.is_tensor(z):
            raise TypeError(f"'z' is not a tensor in latent file: {latent_path}")
        z = z.to(self.dtype)

        # Saved latents are [1, 8, 256, 16]; squeeze singleton batch for DataLoader stacking.
        if z.ndim == 4 and z.shape[0] == 1:
            z = z.squeeze(0)

        genre_id = int(obj["genre_id"])

        if not self.return_metadata:
            return z, genre_id

        metadata = {
            "latent_path": str(latent_path),
            "track_id": int(row["track_id"]),
            "slice_id": int(row["slice_id"]),
            "split": row["split"],
        }
        return z, genre_id, metadata


def get_dataloader(
    manifest_path: str,
    split: str,
    batch_size: int,
    *,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    dtype: torch.dtype = torch.float32,
    return_metadata: bool = False,
    root_dir: Optional[str] = None,
) -> DataLoader:
    """Create a DataLoader for latent training/eval."""
    dataset = LatentDataset(
        manifest_path=manifest_path,
        split=split,
        dtype=dtype,
        return_metadata=return_metadata,
        root_dir=root_dir,
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def load_one_batch(
    manifest_path: str,
    split: str,
    batch_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper for quick smoke tests."""
    loader = get_dataloader(
        manifest_path=manifest_path,
        split=split,
        batch_size=batch_size,
        shuffle=False,
    )
    return next(iter(loader))
