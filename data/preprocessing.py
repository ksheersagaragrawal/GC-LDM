import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import librosa
from diffusers import AutoencoderKL
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import warnings

#Ignore warnings for clean output
warnings.filterwarnings("ignore")

FMA_AUDIO_DIR = "fma_small"
FMA_METADATA_DIR = "fma_metadata"
OUTPUT_DIR = "processed_latents"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("The device is {}".format(device))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# FMA Small Dataset
class FMASmallDataset(Dataset):
    def __init__(self, audio_dir, metadata_dir):
        self.audio_dir = audio_dir
        
        print("Loading FMA metadata...")
        try:
            tracks = pd.read_csv(
                os.path.join(metadata_dir, 'tracks.csv'), 
                index_col=0, header=[0, 1]
            )
        except FileNotFoundError:
            raise RuntimeError("Could not find tracks.csv.")

        small = tracks[tracks[('set', 'subset')] == 'small']
        self.metadata = small['track']['genre_top']
        
        # Create Mapping
        self.unique_genres = sorted(self.metadata.dropna().unique())
        self.class_to_idx = {genre: idx for idx, genre in enumerate(self.unique_genres)}
        
        self.data_list = []
        for track_id, genre_str in self.metadata.items():
            if genre_str in self.class_to_idx:
                self.data_list.append((track_id, self.class_to_idx[genre_str]))
                
        print(f"Prepared {len(self.data_list)} samples.")
        print(f"Genres map: {self.class_to_idx}")

    def __len__(self):
        return len(self.data_list)

    def get_audio_path(self, track_id):
        tid_str = '{:06d}'.format(track_id)
        return os.path.join(self.audio_dir, tid_str[:3], tid_str + '.mp3')

    def __getitem__(self, idx):
        track_id, genre_idx = self.data_list[idx]
        audio_path = self.get_audio_path(track_id)
        
        try:
            # Load full 30s
            audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            return None, None, None, None # Return 4 Nones

        waveform = torch.from_numpy(audio_array).unsqueeze(0)
        return waveform, 16000, genre_idx, track_id

# VAE huggingface
print("Loading AudioLDM VAE...")
vae = AutoencoderKL.from_pretrained("cvssp/audioldm-s-full-v2", subfolder="vae").to(device)
vae.eval()
print("VAE configs are: ")
print(vae.config)

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, win_length=1024, hop_length=160,
    f_min=0, f_max=8000, n_mels=64, power=1.0, 
    norm="slaney", mel_scale="htk"
).to(device)

def preprocess_chunk(chunk_waveform):
    # Input: [1, Time]
    chunk_waveform = chunk_waveform.to(device)

    # STRICT 10s Padding/Cropping
    target_len = 160000 
    if chunk_waveform.shape[1] > target_len:
        chunk_waveform = chunk_waveform[:, :target_len]
    elif chunk_waveform.shape[1] < target_len:
        pad_amount = target_len - chunk_waveform.shape[1]
        chunk_waveform = torch.nn.functional.pad(chunk_waveform, (0, pad_amount))
        
    mel = mel_transform(chunk_waveform)
    log_mel = torch.log(torch.clamp(mel, min=1e-5))
    norm_mel = (log_mel - (-5.0)) / 5.0

    if norm_mel.shape[2] < 1024:
        norm_mel = torch.nn.functional.pad(norm_mel, (0, 1024 - norm_mel.shape[2]))
    elif norm_mel.shape[2] > 1024:
        norm_mel = norm_mel[:, :, :1024]
    
    # Add Batch Dimension [1, 64, 1024] -> [1, 1, 64, 1024]
    return norm_mel.unsqueeze(0) 

dataset = FMASmallDataset(FMA_AUDIO_DIR, FMA_METADATA_DIR)

# Save Genre Map
with open(os.path.join(OUTPUT_DIR, "genre_mapping.json"), "w") as f:
    json.dump(dataset.class_to_idx, f)

SCALING_FACTOR = vae.config.scaling_factor
CHUNK_SIZE = 160000 

print("Starting processing...")
with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        try:
            waveform, sr, genre_id, track_id = dataset[i]
            
            if waveform is None:
                continue
            
            total_samples = waveform.shape[1]
            
            for slice_idx in range(3):
                start = slice_idx * CHUNK_SIZE
                end = start + CHUNK_SIZE
                
                # Minimum 9.9 seconds of audio required for safety
                available_samples = total_samples - start
                if available_samples >= (CHUNK_SIZE - 1600):  # 9.9s minimum
                    actual_end = min(end, total_samples)
                    chunk = waveform[:, start:actual_end]
                    
                    # 1. Preprocess (Get 4D tensor)
                    spectrogram = preprocess_chunk(chunk)
                    
                    # 2. Encode
                    posterior = vae.encode(spectrogram).latent_dist
                    z = posterior.mode() * SCALING_FACTOR
                    
                    # 3. Save
                    save_name = f"{track_id}_{slice_idx}.pt"
                    torch.save({
                        'z': z.cpu().half(),      
                        'genre_id': genre_id     
                    }, os.path.join(OUTPUT_DIR, save_name))
            
        except Exception as e:
            print(f"Error on index {i}: {e}")

print("Done!")