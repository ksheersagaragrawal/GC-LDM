# GC-LDM
Genre-conditioned music generation using latent diffusion models. This project generates original audio by diffusing compressed Mel-spectrogram latents conditioned on genre embeddings, then decoding them into waveforms. Focused on controllability, efficiency, and reproducible research.

## Project layout
- `/GC-LDM/data/preprocessing.py`  
  Build VAE latents from raw audio + metadata.
- `/GC-LDM/scripts/build_latent_manifest.py`  
  Build deterministic track-level splits and manifest/index for latent training.
- `/GC-LDM/scripts/latent_dataset.py`  
  Training dataset and dataloader utilities over manifest files.
- `/GC-LDM/scripts/train_genre_diffusion.py`  
  End-to-end diffusion training with validation, CFG dropout, and checkpointing.
- `/GC-LDM/scripts/sample_genre_diffusion.py`  
  Class-conditioned sampling with CFG and optional audio export (VAE decode + vocoder).
- `/GC-LDM/scripts/evaluate_genre_diffusion.py`  
  Validation loss evaluation plus optional genre-consistency proxy and FAD.
- `/GC-LDM/scripts/sanity_test.py`  
  Single-file reconstruction sanity check (audio -> log-mel -> VAE -> vocoder).
- `/GC-LDM/scripts/decode_and_vocode.py`  
  Decode saved latents back to waveform.

## Quick run
Install deps in your env first:
```bash
pip install -r requirements.txt
```

Sanity test:
```bash
python scripts/sanity_test.py
```

Decode one latent with AudioLDM vocoder:
```bash
python scripts/decode_and_vocode.py \
  --latents data/processed_latents/100478_0.pt \
  --outdir recon_audio \
  --max_files 1 \
  --use_hifigan
```

## Build train/val/test manifest for latents
```bash
python scripts/build_latent_manifest.py \
  --latents_dir data/processed_latents \
  --seed 42 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1
```

This generates:
- `data/processed_latents/manifest.csv`
- `data/processed_latents/splits.json`

## Dataloader quick check
```bash
python - <<'PY'
from scripts.latent_dataset import load_one_batch

z, genre_id = load_one_batch("data/processed_latents/manifest.csv", split="train", batch_size=4)
print(z.shape)        # expected: [B, 8, 256, 16]
print(genre_id.shape) # expected: [B]
PY
```

## Train diffusion model
```bash
python scripts/train_genre_diffusion.py \
  --manifest data/processed_latents/manifest.csv \
  --genre_mapping data/processed_latents/genre_mapping.json \
  --batch_size 4 \
  --epochs 10 \
  --val_every_steps 500 \
  --save_every_steps 500 \
  --cfg_dropout_prob 0.1
```

## Sample with CFG and export audio
```bash
python scripts/sample_genre_diffusion.py \
  --checkpoint runs/train_genre_diffusion/<run_name>/checkpoints/latest.pt \
  --genre_mapping data/processed_latents/genre_mapping.json \
  --genre_ids 0,3,7 \
  --num_samples_per_genre 2 \
  --cfg_scale 3.5 \
  --num_steps 50 \
  --use_hifigan
```

## Evaluate checkpoint
```bash
python scripts/evaluate_genre_diffusion.py \
  --checkpoint runs/train_genre_diffusion/<run_name>/checkpoints/latest.pt \
  --manifest data/processed_latents/manifest.csv \
  --genre_mapping data/processed_latents/genre_mapping.json \
  --split val \
  --max_val_batches 100
```
