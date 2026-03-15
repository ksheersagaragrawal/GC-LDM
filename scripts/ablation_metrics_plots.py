#!/usr/bin/env python3
"""Compute ablation metrics and generate plots.

- Uses log-mel embeddings + Frechet distance as FAD proxy (fast, no hub download).
- Uses AST classifier only to split reference audio into genre folders.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import librosa
from transformers import AutoModelForAudioClassification, AutoProcessor
from scipy.linalg import sqrtm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
GENRES = list(GENRE_KEYWORDS.keys())
DEFAULT_AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


# -------------------------------
# Helpers
# -------------------------------

def resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def parse_combo(name: str) -> Tuple[float, int]:
    cfg_s, steps_s = name.split("_")
    return float(cfg_s), int(steps_s)


def iter_combo_dirs(samples_root: Path) -> List[Path]:
    combos = []
    for p in samples_root.iterdir():
        if not p.is_dir():
            continue
        try:
            parse_combo(p.name)
        except Exception:
            continue
        combos.append(p)
    return sorted(combos)


def list_audio_files(root: Path) -> List[Path]:
    exts = {".wav", ".mp3", ".flac", ".ogg"}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def build_flat_dir(src_files: List[Path], out_dir: Path):
    if out_dir.exists():
        return
    ensure_dir(out_dir)
    for idx, src in enumerate(src_files):
        ext = src.suffix.lower()
        dst = out_dir / f"ref_{idx:05d}{ext}"
        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def limit_files(files: List[Path], max_files: int) -> List[Path]:
    if max_files <= 0 or len(files) <= max_files:
        return files
    return files[:max_files]


# -------------------------------
# AST classifier helpers (for ref split)
# -------------------------------

def build_label_map(id2label: Dict[int, str]) -> Dict[str, List[int]]:
    label_map: Dict[str, List[int]] = {g: [] for g in GENRE_KEYWORDS}
    for idx, label in id2label.items():
        lab = str(label).lower()
        for genre, keys in GENRE_KEYWORDS.items():
            if any(k in lab for k in keys):
                label_map[genre].append(int(idx))
    return label_map


def predict_genre(logits: torch.Tensor, label_map: Dict[str, List[int]], id2label: Dict[int, str]) -> str:
    best_genre = "Unknown"
    best_score = None
    for genre, indices in label_map.items():
        if not indices:
            continue
        score = torch.max(logits[indices]).item()
        if best_score is None or score > best_score:
            best_score = score
            best_genre = genre

    if best_genre == "Unknown":
        top_idx = int(torch.argmax(logits).item())
        top_label = str(id2label.get(top_idx, "")).lower()
        for genre, keys in GENRE_KEYWORDS.items():
            if any(k in top_label for k in keys):
                return genre
    return best_genre


def split_reference_by_genre(
    reference_audio_dir: Path,
    out_dir: Path,
    processor,
    model,
    device: str,
):
    if out_dir.exists() and any(out_dir.iterdir()):
        return

    ensure_dir(out_dir)
    for g in GENRES:
        ensure_dir(out_dir / g)

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label_map = build_label_map(id2label)

    ref_files = list_audio_files(reference_audio_dir)
    for idx, audio_path in enumerate(ref_files):
        y, _ = librosa.load(audio_path, sr=16000, mono=True)
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(0)
        genre = predict_genre(logits, label_map, id2label)
        if genre == "Unknown":
            continue
        dst = out_dir / genre / f"ref_{idx:05d}{audio_path.suffix.lower()}"
        try:
            os.symlink(audio_path, dst)
        except Exception:
            shutil.copy2(audio_path, dst)


# -------------------------------
# Diversity metrics
# -------------------------------

def load_latents_from_combo(combo_dir: Path) -> Dict[str, List[torch.Tensor]]:
    by_genre: Dict[str, List[torch.Tensor]] = {}
    for gdir in sorted(combo_dir.glob("genre_*_*")):
        genre_name = gdir.name.split("_")[1]
        by_genre.setdefault(genre_name, [])
        for pt in sorted(gdir.glob("*.pt")):
            obj = torch.load(pt, map_location="cpu")
            z = obj.get("z")
            if z is None:
                continue
            if z.ndim == 4 and z.shape[0] == 1:
                z = z.squeeze(0)
            by_genre[genre_name].append(z.reshape(-1).float())
    return by_genre


def compute_diversity_stats(by_genre: Dict[str, List[torch.Tensor]]) -> Dict[str, object]:
    centroids = {}
    intra_dists = []
    for g, vecs in by_genre.items():
        if not vecs:
            continue
        stack = torch.stack(vecs, dim=0)
        centroid = stack.mean(dim=0)
        centroids[g] = centroid
        dists = torch.norm(stack - centroid, dim=1).cpu().numpy().tolist()
        intra_dists.extend(dists)

    inter_dists = []
    gnames = sorted(centroids.keys())
    for i in range(len(gnames)):
        for j in range(i + 1, len(gnames)):
            d = torch.norm(centroids[gnames[i]] - centroids[gnames[j]], p=2).item()
            inter_dists.append(d)

    return {
        "intra_dists": intra_dists,
        "inter_dists": inter_dists,
        "intra_mean": float(np.mean(intra_dists)) if intra_dists else 0.0,
        "inter_mean": float(np.mean(inter_dists)) if inter_dists else 0.0,
    }


# -------------------------------
# Log-mel FAD proxy
# -------------------------------

def logmel_embedding(audio_path: Path, sr: int = 16000, n_mels: int = 64) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, power=2.0)
    logmel = librosa.power_to_db(mel + 1e-6)
    return logmel.mean(axis=1)


def compute_logmel_embeddings(files: List[Path]) -> np.ndarray:
    embs = [logmel_embedding(p) for p in files]
    return np.stack(embs, axis=0) if embs else np.zeros((0, 64))


def frechet_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    if emb1.shape[0] < 2 or emb2.shape[0] < 2:
        return float("nan")
    mu1 = np.mean(emb1, axis=0)
    mu2 = np.mean(emb2, axis=0)
    sigma1 = np.cov(emb1, rowvar=False)
    sigma2 = np.cov(emb2, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


# -------------------------------
# Plot helpers
# -------------------------------

def plot_train_val_loss(train_log: Path, val_log: Path, out_path: Path):
    train_rows = load_jsonl(train_log)
    val_rows = load_jsonl(val_log)
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    plt.figure(figsize=(7, 4))
    plt.plot(train_df["global_step"], train_df["train_loss"], label="train_loss", alpha=0.7)
    plt.plot(val_df["global_step"], val_df["val_loss"], label="val_loss", alpha=0.9)
    plt.xlabel("Global step")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss vs Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_line_grouped(df: pd.DataFrame, x: str, y: str, hue: str, title: str, out_path: Path):
    plt.figure(figsize=(7, 4))
    for key, grp in df.groupby(hue):
        g = grp.sort_values(x)
        plt.plot(g[x], g[y], marker="o", label=str(key))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend(title=hue)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path):
    plt.figure(figsize=(6, 4))
    plt.scatter(df[x], df[y], s=40)
    for _, row in df.iterrows():
        plt.annotate(row["combo"], (row[x], row[y]), fontsize=7, alpha=0.7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_bar(labels: List[str], values: List[float], title: str, out_path: Path, ylabel: str):
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_boxplot(data: List[List[float]], labels: List[str], title: str, out_path: Path, ylabel: str):
    plt.figure(figsize=(5, 4))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -------------------------------
# Main
# -------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute ablation metrics + plots")
    p.add_argument("--samples_root", default="runs/samples/ablation_output_generated")
    p.add_argument("--classifier_analysis", default=None)
    p.add_argument("--train_log", default="runs/train_genre_diffusion/colab_run_01/train_log.jsonl")
    p.add_argument("--val_log", default="runs/train_genre_diffusion/colab_run_01/val_log.jsonl")
    p.add_argument("--reference_audio", default="data/100")
    p.add_argument("--out_metrics", default="results/metrics")
    p.add_argument("--out_plots", default="results/plots")
    p.add_argument("--ast_model", default=DEFAULT_AST_MODEL)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_ref_files", type=int, default=80)
    p.add_argument("--max_gen_files", type=int, default=40)
    p.add_argument("--max_files_per_genre", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    samples_root = resolve_path(args.samples_root)
    analysis_dir = resolve_path(args.classifier_analysis) if args.classifier_analysis else (samples_root / "analysis")
    out_metrics = resolve_path(args.out_metrics)
    out_plots = resolve_path(args.out_plots)
    ensure_dir(out_metrics)
    ensure_dir(out_plots)

    # 1) Train/val loss plot
    plot_train_val_loss(resolve_path(args.train_log), resolve_path(args.val_log), out_plots / "train_val_loss.png")

    # 2) Classifier summary
    summary_path = analysis_dir / "summary.csv"
    summary_df = pd.read_csv(summary_path)
    summary_df[["cfg", "steps"]] = summary_df["combo"].apply(lambda x: pd.Series(parse_combo(x)))

    # Per-genre accuracy + precision/recall/F1
    per_genre_rows = []
    for combo in summary_df["combo"].tolist():
        metrics_path = analysis_dir / combo / "metrics.json"
        cm_path = analysis_dir / combo / "confusion_matrix.csv"
        if not metrics_path.exists() or not cm_path.exists():
            continue
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        cm = pd.read_csv(cm_path, index_col=0)
        for g in GENRES:
            tp = cm.loc[g, g] if g in cm.index and g in cm.columns else 0
            prec = float(tp / cm[g].sum()) if g in cm.columns and cm[g].sum() > 0 else 0.0
            rec = float(tp / cm.loc[g].sum()) if g in cm.index and cm.loc[g].sum() > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            per_genre_rows.append({
                "combo": combo,
                "genre": g,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "accuracy": metrics.get("per_genre_accuracy", {}).get(g, 0.0),
            })
    per_genre_df = pd.DataFrame(per_genre_rows)
    per_genre_df.to_csv(out_metrics / "per_genre_metrics.csv", index=False)

    # 3) AST model for reference split
    device = args.device
    processor = AutoProcessor.from_pretrained(args.ast_model)
    model = AutoModelForAudioClassification.from_pretrained(args.ast_model).to(device)
    model.eval()

    reference_audio = resolve_path(args.reference_audio)
    reference_all_dir = out_metrics / "reference_all"
    ref_files = limit_files(list_audio_files(reference_audio), args.max_ref_files)
    build_flat_dir(ref_files, reference_all_dir)

    reference_by_genre = out_metrics / "reference_by_genre"
    split_reference_by_genre(reference_audio, reference_by_genre, processor, model, device)

    # 4) Log-mel FAD proxy
    ref_all_emb = compute_logmel_embeddings(limit_files(list_audio_files(reference_all_dir), args.max_ref_files))
    ref_by_genre_emb = {}
    for g in GENRES:
        gdir = reference_by_genre / g
        if gdir.exists() and any(gdir.iterdir()):
            ref_by_genre_emb[g] = compute_logmel_embeddings(
                limit_files(list_audio_files(gdir), args.max_files_per_genre)
            )

    fad_summary_rows = []
    for combo_dir in iter_combo_dirs(samples_root):
        combo = combo_dir.name
        cfg, steps = parse_combo(combo)
        gen_all = limit_files(list_audio_files(combo_dir), args.max_gen_files)
        gen_all_emb = compute_logmel_embeddings(gen_all)
        score = frechet_distance(ref_all_emb, gen_all_emb)
        fad_summary_rows.append({"combo": combo, "cfg": cfg, "steps": steps, "fad": score})

    fad_summary_df = pd.DataFrame(fad_summary_rows)
    fad_summary_df.to_csv(out_metrics / "fad_summary.csv", index=False)

    # per-genre FAD for best (lowest) combo
    fad_per_genre_rows = []
    if not fad_summary_df.empty:
        best_combo = fad_summary_df.sort_values("fad").iloc[0]["combo"]
        best_dir = samples_root / best_combo
        for g in GENRES:
            gen_dir = next(best_dir.glob(f"genre_{g}_*"), None)
            if gen_dir is None:
                continue
            gen_emb = compute_logmel_embeddings(
                limit_files(list_audio_files(gen_dir), args.max_files_per_genre)
            )
            ref_emb = ref_by_genre_emb.get(g)
            if ref_emb is None or ref_emb.shape[0] < 2 or gen_emb.shape[0] < 2:
                continue
            score = frechet_distance(ref_emb, gen_emb)
            fad_per_genre_rows.append({"combo": best_combo, "genre": g, "fad": score})
        if fad_per_genre_rows:
            pd.DataFrame(fad_per_genre_rows).to_csv(out_metrics / "fad_per_genre_best.csv", index=False)

    # 5) Diversity metrics
    diversity_rows = []
    diversity_dist_by_combo = {}
    for combo_dir in iter_combo_dirs(samples_root):
        combo = combo_dir.name
        cfg, steps = parse_combo(combo)
        by_genre = load_latents_from_combo(combo_dir)
        stats = compute_diversity_stats(by_genre)
        diversity_rows.append({
            "combo": combo,
            "cfg": cfg,
            "steps": steps,
            "intra_mean": stats["intra_mean"],
            "inter_mean": stats["inter_mean"],
        })
        diversity_dist_by_combo[combo] = stats

    diversity_df = pd.DataFrame(diversity_rows)
    diversity_df.to_csv(out_metrics / "diversity_summary.csv", index=False)

    # 6) Plots
    plot_line_grouped(
        summary_df,
        x="steps",
        y="overall_accuracy",
        hue="cfg",
        title="Genre Accuracy vs Inference Steps",
        out_path=out_plots / "genre_accuracy_vs_steps.png",
    )
    plot_line_grouped(
        summary_df,
        x="cfg",
        y="overall_accuracy",
        hue="steps",
        title="Genre Accuracy vs CFG Scale",
        out_path=out_plots / "genre_accuracy_vs_cfg.png",
    )

    # Confusion matrix heatmap (best combo by accuracy)
    best_acc_combo = summary_df.sort_values("overall_accuracy", ascending=False).iloc[0]["combo"]
    cm_png = analysis_dir / best_acc_combo / "confusion_matrix.png"
    if cm_png.exists():
        shutil.copy2(cm_png, out_plots / "confusion_matrix_best.png")

    # Per-genre accuracy bar (best combo by accuracy)
    best_genre_df = per_genre_df[per_genre_df["combo"] == best_acc_combo]
    if not best_genre_df.empty:
        plot_bar(
            best_genre_df["genre"].tolist(),
            best_genre_df["recall"].tolist(),
            title=f"Per-Genre Recall (Best Combo {best_acc_combo})",
            out_path=out_plots / "per_genre_accuracy_best.png",
            ylabel="Recall",
        )
        best_genre_df.to_csv(out_metrics / "per_genre_metrics_best.csv", index=False)

    # FAD plots
    plot_line_grouped(
        fad_summary_df,
        x="steps",
        y="fad",
        hue="cfg",
        title="FAD (log-mel) vs Inference Steps",
        out_path=out_plots / "fad_vs_steps.png",
    )
    plot_line_grouped(
        fad_summary_df,
        x="cfg",
        y="fad",
        hue="steps",
        title="FAD (log-mel) vs CFG Scale",
        out_path=out_plots / "fad_vs_cfg.png",
    )

    # Quality vs control (FAD vs accuracy)
    merged = pd.merge(summary_df, fad_summary_df, on=["combo", "cfg", "steps"], how="inner")
    if not merged.empty:
        plot_scatter(
            merged,
            x="fad",
            y="overall_accuracy",
            title="Quality vs Control (FAD vs Genre Accuracy)",
            out_path=out_plots / "quality_vs_control.png",
        )

    # Per-genre FAD bar for best combo by FAD
    fad_per_genre_path = out_metrics / "fad_per_genre_best.csv"
    if fad_per_genre_path.exists():
        per_g = pd.read_csv(fad_per_genre_path)
        plot_bar(
            per_g["genre"].tolist(),
            per_g["fad"].tolist(),
            title=f"Per-Genre FAD (Best Combo {per_g['combo'].iloc[0]})",
            out_path=out_plots / "fad_per_genre_best.png",
            ylabel="FAD (log-mel)",
        )

    # Diversity plot (boxplot for best combo by accuracy)
    best_combo = best_acc_combo
    if best_combo in diversity_dist_by_combo:
        stats = diversity_dist_by_combo[best_combo]
        plot_boxplot(
            [stats["intra_dists"], stats["inter_dists"]],
            ["intra", "inter"],
            title=f"Diversity (Best Combo {best_combo})",
            out_path=out_plots / "diversity_intra_inter.png",
            ylabel="L2 distance",
        )

    # Quality-speed tradeoff: steps vs FAD (mean over cfg)
    mean_fad = fad_summary_df.groupby("steps", as_index=False)["fad"].mean()
    plt.figure(figsize=(6, 4))
    plt.plot(mean_fad["steps"], mean_fad["fad"], marker="o", label="FAD (log-mel)")
    rel_runtime = mean_fad["steps"] / mean_fad["steps"].max()
    plt.plot(mean_fad["steps"], rel_runtime, marker="o", label="relative runtime (proxy)")
    plt.xlabel("Inference steps")
    plt.ylabel("Value")
    plt.title("Quality-Speed Tradeoff (FAD + Runtime Proxy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plots / "quality_speed_tradeoff.png", dpi=160)
    plt.close()

    print(f"Saved metrics to {out_metrics}")
    print(f"Saved plots to {out_plots}")


if __name__ == "__main__":
    main()
