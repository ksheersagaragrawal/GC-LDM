# GC-LDM
Genre-conditioned music generation using latent diffusion models. This project generates original audio by diffusing compressed Mel-spectrogram latents conditioned on genre embeddings, then decoding them into waveforms. Focused on controllability, efficiency, and reproducible research.

## Project layout
- `/GC-LDM/data/preprocessing.py`  
  Build VAE latents from raw audio + metadata.
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
