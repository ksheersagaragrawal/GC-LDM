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

class Dataset(Dataset):
    def __init__(self, audio_directory, metadata_directory):
        self.audio_directory = audio_directory
        
        tracks_csv = pd.read_csv(
                os.path.join(metadata_directory, 'tracks.csv'), 
                index_col=0, header=[0, 1]
            )
        
        #This csv file contains two level hierarchy of column names
        tracks_set_columns = tracks_csv['set']
        small_tracks = tracks_csv[tracks_set_columns['subset'] == 'small']
        self.metadata = small_tracks['track']['genre_top']
        
        unique_genres = sorted(self.metadata.dropna().unique())
        self.genre_to_index = {}
        index = 0
        for genre in unique_genres:
            self.genre_to_index[genre] = index
            index = index + 1
        
        self.data_list = []
        for track_id, genre_str in self.metadata.items():
            if genre_str in self.genre_to_index:
                self.data_list.append((track_id, self.genre_to_index[genre_str]))
                
        print("Prepared {} samples".format(len(self.data_list)))
        print("Genres map: {}".format(self.genre_to_index))

    def __len__(self):
        #This function returns the length of the data list which is used to get the length of dataloader
        return len(self.data_list)

    def get_audio_path(self, track_id):
        #This function returns the entire audio path folder/audio_file
        track_id_string = '{:06d}'.format(track_id)
        folder_name = track_id_string[:3]
        return os.path.join(self.audio_directory, folder_name, track_id_string + '.mp3')

    def __getitem__(self, index):
        #This function is used to get a particular track
        track_id, genre_index = self.data_list[index]
        audio_path = self.get_audio_path(track_id)
        
        try:
            # Load full 30s, 16000 is the default sample rate used from huggingface AudioLDM
            audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            return None, None, None, None # Return 4 Nones

        waveform = torch.from_numpy(audio_array).unsqueeze(0)
        sample_rate = 16000 #This is the default present in huggingface website for AudioLDM
        return waveform, sample_rate, genre_index, track_id

def preprocess_chunk(chunk_waveform):
    chunk_waveform = chunk_waveform.to(device)

    # Padding the time to 10 seconds for each track
    sample_rate = 16000
    required_sample_length = sample_rate * 10 #Multiplying with 10 for 10 seconds 
    current_sample = chunk_waveform.shape[1]
    if current_sample >= required_sample_length:
        chunk_waveform = chunk_waveform[:, :required_sample_length]
    else:
        required_padding = required_sample_length - current_sample
        chunk_waveform = torch.nn.functional.pad(chunk_waveform, (0, required_padding))

    mel = mel_spectrogram(chunk_waveform)
    log_mel = torch.log(torch.clamp(mel, min=1e-5))
    #Similar to AudioLDM
    norm_mel = (log_mel - (-5.0)) / 5.0

    no_of_frames = norm_mel.shape[2]
    if no_of_frames >=1024:
        norm_mel = norm_mel[:, :, :1024]
    else:
        required_padding = 1024 - no_of_frames
        norm_mel = torch.nn.functional.pad(norm_mel, (0, required_padding))

    # Add Batch Dimension [1, 64, 1024] -> [1, 1, 64, 1024]
    return norm_mel.unsqueeze(0)

if __name__ == '__main__':

    audio_directory = "fma_small"
    metadata_directory = "fma_metadata"
    output_directory = "processed_latents"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("The device is {}".format(device))

    os.makedirs(output_directory, exist_ok=True)

    #Load the dataset
    dataset = Dataset(audio_directory, metadata_directory)

    # Save Genre Map
    with open(os.path.join(output_directory, "genre_mapping.json"), "w") as f:
        json.dump(dataset.genre_to_index, f)

    print("Loading AudioLDM VAE from huggingface")
    variational_auto_encoder = AutoencoderKL.from_pretrained("cvssp/audioldm-s-full-v2", subfolder="vae").to(device)
    #We need to use the autoencoder for preprocessing and not training, hence we need to change it to eval mode
    variational_auto_encoder.eval()
    #Setting the scaling factor from the vae's scaling factor from configs
    SCALING_FACTOR = variational_auto_encoder.config.scaling_factor

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, win_length=1024, hop_length=160,
        f_min=0, f_max=8000, n_mels=64, power=1.0, 
        norm="slaney", mel_scale="htk"
    ).to(device)

    zero_point_one_second = 1600
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            try:
                waveform, sampling_rate, genre_id, track_id = dataset[i]
                
                if waveform is None:
                    continue

                total_samples = waveform.shape[1]
    
                for slice_index in range(3):
                    start_index = slice_index * 160000
                    end_index = start_index + 16000
            
                    # Minimum 9.9 seconds of audio required for safety
                    available_samples = total_samples - start_index
                    if available_samples >= (160000 - zero_point_one_second):  # 9.9s minimum
                        actual_end_index = min(end_index, total_samples)
                        chunk = waveform[:, start_index:actual_end_index]
                    
                        # 1. Preprocess (Get 4D tensor)
                        spectrogram = preprocess_chunk(chunk)
                    
                        # 2. Encode
                        posterior = variational_auto_encoder.encode(spectrogram).latent_dist
                        z = posterior.mode() * SCALING_FACTOR
                    
                        # 3. Save
                        save_name = "{}_{}.pt".format(track_id, slice_index)
                        torch.save({
                            'z': z.cpu().half(),      
                            'genre_id': genre_id     
                        }, os.path.join(output_directory, save_name))
            
            except Exception as e:
                print("Error on index {}: {}".format(i,e))
