"""Build a deterministic track-level split and manifest for latent training data."""

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LATENTS_DIR = PROJECT_ROOT / "data" / "processed_latents"
FILENAME_RE = re.compile(r"^(?P<track_id>\d+)_(?P<slice_id>\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build latent manifest and track-level splits.")
    parser.add_argument("--latents_dir", type=Path, default=DEFAULT_LATENTS_DIR)
    parser.add_argument("--manifest_out", type=Path, default=None)
    parser.add_argument("--splits_out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument(
        "--relative_to",
        type=Path,
        default=PROJECT_ROOT,
        help="Store latent_path relative to this directory in manifest",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float):
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Ratios must be non-negative and train_ratio must be > 0")
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")


def split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    raw = [n * train_ratio, n * val_ratio, n * test_ratio]
    base = [int(math.floor(x)) for x in raw]
    remainder = n - sum(base)
    frac_order = sorted(
        range(3),
        key=lambda idx: (raw[idx] - base[idx]),
        reverse=True,
    )
    for idx in frac_order[:remainder]:
        base[idx] += 1
    return base[0], base[1], base[2]


def build_records(latents_dir: Path) -> Tuple[List[Dict], List[str], Dict[int, int]]:
    records: List[Dict] = []
    errors: List[str] = []
    track_to_genre: Dict[int, int] = {}
    pt_files = sorted(latents_dir.glob("*.pt"))

    for path in pt_files:
        if path.name == "genre_mapping.json":
            continue
        match = FILENAME_RE.match(path.name)
        if not match:
            errors.append(f"unexpected filename format: {path.name}")
            continue

        track_id = int(match.group("track_id"))
        slice_id = int(match.group("slice_id"))

        try:
            obj = torch.load(path, map_location="cpu")
        except Exception as exc:
            errors.append(f"failed to load {path.name}: {exc}")
            continue

        if "z" not in obj or "genre_id" not in obj:
            errors.append(f"missing keys in {path.name}; expected 'z' and 'genre_id'")
            continue

        z = obj["z"]
        genre_id = int(obj["genre_id"])

        if not torch.is_tensor(z):
            errors.append(f"'z' is not a tensor in {path.name}")
            continue

        if track_id in track_to_genre and track_to_genre[track_id] != genre_id:
            errors.append(
                f"inconsistent genre_id for track {track_id}: "
                f"{track_to_genre[track_id]} vs {genre_id}"
            )
            continue
        track_to_genre[track_id] = genre_id

        records.append(
            {
                "path": path,
                "track_id": track_id,
                "slice_id": slice_id,
                "genre_id": genre_id,
                "z_shape": list(z.shape),
            }
        )

    return records, errors, track_to_genre


def track_level_split(
    track_to_genre: Dict[int, int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[int, str]:
    rng = random.Random(seed)
    genre_to_tracks: Dict[int, List[int]] = defaultdict(list)
    for track_id, genre_id in track_to_genre.items():
        genre_to_tracks[genre_id].append(track_id)

    split_by_track: Dict[int, str] = {}
    for genre_id in sorted(genre_to_tracks):
        tracks = sorted(genre_to_tracks[genre_id])
        rng.shuffle(tracks)
        n_train, n_val, n_test = split_counts(
            len(tracks),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        train_ids = tracks[:n_train]
        val_ids = tracks[n_train : n_train + n_val]
        test_ids = tracks[n_train + n_val : n_train + n_val + n_test]
        for tid in train_ids:
            split_by_track[tid] = "train"
        for tid in val_ids:
            split_by_track[tid] = "val"
        for tid in test_ids:
            split_by_track[tid] = "test"

    if len(split_by_track) != len(track_to_genre):
        raise RuntimeError("Track split is incomplete; some tracks were not assigned")
    return split_by_track


def validate_genre_ids(records: List[Dict], latents_dir: Path, errors: List[str]):
    mapping_path = latents_dir / "genre_mapping.json"
    if not mapping_path.exists():
        errors.append("genre_mapping.json is missing")
        return
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    valid_ids = set(mapping.values())
    for row in records:
        if row["genre_id"] not in valid_ids:
            errors.append(
                f"genre_id {row['genre_id']} in {row['path'].name} not in genre_mapping.json"
            )


def write_manifest(
    records: List[Dict],
    split_by_track: Dict[int, str],
    manifest_out: Path,
    relative_to: Path,
):
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["latent_path", "track_id", "slice_id", "genre_id", "split"],
        )
        writer.writeheader()
        for row in sorted(records, key=lambda x: (x["track_id"], x["slice_id"])):
            try:
                latent_path = row["path"].resolve().relative_to(relative_to.resolve())
            except ValueError:
                latent_path = row["path"].resolve()
            writer.writerow(
                {
                    "latent_path": str(latent_path),
                    "track_id": row["track_id"],
                    "slice_id": row["slice_id"],
                    "genre_id": row["genre_id"],
                    "split": split_by_track[row["track_id"]],
                }
            )


def write_splits_json(
    records: List[Dict],
    split_by_track: Dict[int, str],
    errors: List[str],
    splits_out: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    tracks_by_split: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for track_id, split in split_by_track.items():
        tracks_by_split[split].append(track_id)
    for split in tracks_by_split:
        tracks_by_split[split] = sorted(tracks_by_split[split])

    latent_counts = Counter()
    track_counts = {k: len(v) for k, v in tracks_by_split.items()}
    genre_hist: Dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}
    shape_hist: Dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}

    for row in records:
        split = split_by_track[row["track_id"]]
        latent_counts[split] += 1
        genre_hist[split][str(row["genre_id"])] += 1
        shape_hist[split][str(tuple(row["z_shape"]))] += 1

    payload = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "counts": {
            "total_tracks": len(split_by_track),
            "total_latents": len(records),
            "tracks": track_counts,
            "latents": {k: latent_counts.get(k, 0) for k in ["train", "val", "test"]},
        },
        "tracks_by_split": tracks_by_split,
        "genre_histograms": {k: dict(v) for k, v in genre_hist.items()},
        "latent_shape_histograms": {k: dict(v) for k, v in shape_hist.items()},
        "invalid_files": errors,
    }

    splits_out.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_out, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    latents_dir = args.latents_dir.resolve()
    manifest_out = (args.manifest_out or (latents_dir / "manifest.csv")).resolve()
    splits_out = (args.splits_out or (latents_dir / "splits.json")).resolve()
    relative_to = args.relative_to.resolve()

    if not latents_dir.exists():
        raise FileNotFoundError(f"latents_dir does not exist: {latents_dir}")

    records, errors, track_to_genre = build_records(latents_dir=latents_dir)
    validate_genre_ids(records=records, latents_dir=latents_dir, errors=errors)
    split_by_track = track_level_split(
        track_to_genre=track_to_genre,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    write_manifest(
        records=records,
        split_by_track=split_by_track,
        manifest_out=manifest_out,
        relative_to=relative_to,
    )
    write_splits_json(
        records=records,
        split_by_track=split_by_track,
        errors=errors,
        splits_out=splits_out,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    overlap = (
        set()
        if not split_by_track
        else (
            set.intersection(
                set(t for t, s in split_by_track.items() if s == "train"),
                set(t for t, s in split_by_track.items() if s == "val"),
                set(t for t, s in split_by_track.items() if s == "test"),
            )
        )
    )
    if overlap:
        raise RuntimeError(f"track-level split leakage detected: {len(overlap)} overlapping ids")

    print(f"Latents scanned: {len(records)}")
    print(f"Tracks split: {len(track_to_genre)}")
    print(f"Manifest written: {manifest_out}")
    print(f"Splits written: {splits_out}")
    print(f"Invalid files flagged: {len(errors)}")


if __name__ == "__main__":
    main()
