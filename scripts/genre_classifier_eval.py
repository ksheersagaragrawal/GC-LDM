"""Evaluate genre consistency on generated audio using a pretrained classifier.

Default: use AudioSet AST classifier and map its labels to our 8 genres by keyword.
This is heuristic and intended as a sanity-check, not a perfect genre classifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

from transformers import AutoModelForAudioClassification, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

GENRE_KEYWORDS = {
    "Electronic": ["electronic", "edm", "techno", "house", "synth"],
    "Experimental": ["experimental", "ambient", "noise", "drone"],
    "Folk": ["folk", "acoustic"],
    "Hip-Hop": ["hip hop", "hip-hop", "rap"],
    "Instrumental": ["instrumental"],
    "International": ["world", "international", "latin", "afro", "reggae"],
    "Pop": ["pop"],
    "Rock": ["rock", "metal", "punk"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genre classifier eval over generated samples.")
    parser.add_argument(
        "--samples_root",
        type=str,
        default="runs/samples/ablation_output_generated",
        help="Root containing ablation folders (e.g., 3.5_10/genre_Rock_7/*.wav)",
    )
    parser.add_argument("--out_dir", type=str, default=None, help="Default: <samples_root>/analysis")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_files_per_genre", type=int, default=0, help="0 means no cap")
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def iter_wav_files(root: Path, max_files_per_genre: int = 0) -> Iterable[Tuple[str, Path]]:
    for genre_dir in sorted(root.glob("genre_*_*")):
        wavs = sorted(genre_dir.glob("*.wav"))
        if max_files_per_genre > 0:
            wavs = wavs[: max_files_per_genre]
        for wav in wavs:
            yield genre_dir.name, wav


def build_label_map(id2label: Dict[int, str]) -> Dict[str, List[int]]:
    label_map: Dict[str, List[int]] = {g: [] for g in GENRE_KEYWORDS}
    for idx, label in id2label.items():
        lab = str(label).lower()
        for genre, keys in GENRE_KEYWORDS.items():
            if any(k in lab for k in keys):
                label_map[genre].append(int(idx))
    return label_map


def predict_genre(
    logits: torch.Tensor,
    label_map: Dict[str, List[int]],
    id2label: Dict[int, str],
) -> str:
    best_genre = "Unknown"
    best_score = None
    for genre, indices in label_map.items():
        if not indices:
            continue
        score = torch.max(logits[indices]).item()
        if best_score is None or score > best_score:
            best_score = score
            best_genre = genre

    # Fallback: map the top label by keywords if none matched.
    if best_genre == "Unknown":
        top_idx = int(torch.argmax(logits).item())
        top_label = str(id2label.get(top_idx, "")).lower()
        for genre, keys in GENRE_KEYWORDS.items():
            if any(k in top_label for k in keys):
                return genre
    return best_genre


def parse_true_genre(genre_dir_name: str) -> str:
    # Expect format: genre_<Name>_<id>
    parts = genre_dir_name.split("_")
    if len(parts) >= 3:
        return parts[1]
    return genre_dir_name


def evaluate_folder(
    folder: Path,
    model,
    processor,
    label_map: Dict[str, List[int]],
    id2label: Dict[int, str],
    device: torch.device,
    max_files_per_genre: int = 0,
) -> pd.DataFrame:
    rows = []
    for genre_dir, wav_path in iter_wav_files(folder, max_files_per_genre=max_files_per_genre):
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(0)
        pred = predict_genre(logits, label_map, id2label)
        rows.append(
            {
                "combo": folder.name,
                "true_genre_dir": genre_dir,
                "true_genre": parse_true_genre(genre_dir),
                "pred_genre": pred,
                "wav_path": str(wav_path),
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix(df: pd.DataFrame, genres: List[str]) -> pd.DataFrame:
    labels = genres + ["Unknown"]
    cm = pd.DataFrame(0, index=labels, columns=labels)
    for _, row in df.iterrows():
        true_g = row["true_genre"]
        pred_g = row["pred_genre"] if row["pred_genre"] in labels else "Unknown"
        if true_g not in labels:
            true_g = "Unknown"
        cm.loc[true_g, pred_g] += 1
    return cm


def plot_confusion(cm: pd.DataFrame, out_path: Path):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Genre Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    samples_root = resolve_path(args.samples_root)
    if not samples_root.exists():
        raise FileNotFoundError(f"samples_root not found: {samples_root}")

    out_dir = resolve_path(args.out_dir) if args.out_dir else (samples_root / "analysis").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForAudioClassification.from_pretrained(args.model_id).to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label_map = build_label_map(id2label)

    summary_rows = []
    genres = list(GENRE_KEYWORDS.keys())

    for combo_dir in sorted([p for p in samples_root.iterdir() if p.is_dir()]):
        df = evaluate_folder(
            combo_dir, model, processor, label_map, id2label, device, args.max_files_per_genre
        )
        if df.empty:
            continue
        cm = confusion_matrix(df, genres)
        acc = float(np.trace(cm.values) / max(1, cm.values.sum()))
        per_genre = {}
        for g in genres:
            row_sum = int(cm.loc[g].sum())
            per_genre[g] = float(cm.loc[g, g] / row_sum) if row_sum > 0 else 0.0

        combo_out = out_dir / combo_dir.name
        combo_out.mkdir(parents=True, exist_ok=True)
        df.to_csv(combo_out / "predictions.csv", index=False)
        cm.to_csv(combo_out / "confusion_matrix.csv")
        plot_confusion(cm, combo_out / "confusion_matrix.png")

        metrics = {
            "combo": combo_dir.name,
            "model_id": args.model_id,
            "overall_accuracy": acc,
            "per_genre_accuracy": per_genre,
            "num_samples": int(len(df)),
        }
        with open(combo_out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        summary_rows.append(metrics)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"Saved classifier eval outputs to: {out_dir}")


if __name__ == "__main__":
    main()
