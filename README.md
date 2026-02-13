# GC-LDM
Genre-conditioned music generation using latent diffusion models. This project generates original audio by diffusing compressed Mel-spectrogram latents conditioned on genre embeddings, then decoding them into waveforms. Focused on controllability, efficiency, and reproducible research.

## Project layout
- `/GC-LDM/data/preprocessing.py`  
  Build VAE latents from raw audio + metadata.
- `/GC-LDM/models/gen_conditioning.py`  
  Genre embedding to cross-attention token module (`[B, T, D]`) with null-conditioning support.
- `/GC-LDM/models/genre_diffusion_model.py`  
  Genre-conditioned UNet wrapper with config contract and save/load utilities.
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

## Genre-conditioned model quick usage
```bash
python - <<'PY'
import torch
from models.genre_diffusion_model import GenreConditionedUNet

model = GenreConditionedUNet.from_genre_mapping("data/processed_latents/genre_mapping.json")
z = torch.randn(2, 8, 256, 16)
t = torch.tensor([10, 10], dtype=torch.long)
genre_ids = torch.tensor([0, 7], dtype=torch.long)
eps_pred = model(z, t, genre_ids)
print(eps_pred.shape)

model.save_model("checkpoints/genre_unet.pt")
reloaded = GenreConditionedUNet.load_model("checkpoints/genre_unet.pt")
print(type(reloaded).__name__)
PY
```
