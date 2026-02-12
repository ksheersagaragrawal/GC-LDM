# GC-LDM
Genre-conditioned music generation using latent diffusion.

## Project layout
- `/Users/ksheersagaragrawal/Desktop/WI26/ECE 285/GC-LDM/data/preprocessing.py`  
  Build VAE latents from raw audio + metadata.
- `/Users/ksheersagaragrawal/Desktop/WI26/ECE 285/GC-LDM/scripts/sanity_test.py`  
  Single-file reconstruction sanity check (audio -> log-mel -> VAE -> vocoder).
- `/Users/ksheersagaragrawal/Desktop/WI26/ECE 285/GC-LDM/scripts/decode_and_vocode.py`  
  Decode saved latents back to waveform.

## Important note (fixed mismatch)
The pretrained `cvssp/audioldm-s-full-v2` VAE/vocoder expects:
- raw log-mel input (not mean/std normalized mel)
- VAE layout `(B, 1, T, F)` i.e. `(B, 1, 1024, 64)`

Using normalized mel caused high VAE MSE and poor recon audio.

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

## Smoke test used
```bash
python -m py_compile data/preprocessing.py scripts/sanity_test.py scripts/decode_and_vocode.py
python scripts/sanity_test.py
python scripts/decode_and_vocode.py --latents data/processed_latents/100478_0.pt --outdir sanity_test_output/decode_smoke --max_files 1 --use_hifigan
```
