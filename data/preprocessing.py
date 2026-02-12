import os
import torch
import pandas as pd
import numpy as np
import librosa
from diffusers import AutoencoderKL
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from scipy.signal import get_window
from librosa.util import pad_center
from librosa.filters import mel as librosa_mel_fn


# ── AudioLDM's actual STFT + Mel (from haoheliu/AudioLDM repo) ──
class AudioLDM_STFT(torch.nn.Module):
    def __init__(self, filter_length, hop_length, win_length, window="hann"):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        scale = filter_length / hop_length
        fourier_basis = np.fft.fft(np.eye(filter_length))
        cutoff = int(filter_length / 2 + 1)
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])
        fft_window = get_window(window, win_length, fftbins=True)
        fft_window = pad_center(fft_window, size=filter_length)
        fft_window = torch.from_numpy(fft_window).float()
        forward_basis *= fft_window
        inverse_basis *= fft_window
        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = torch.nn.functional.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)
        forward_transform = torch.nn.functional.conv1d(
            input_data, self.forward_basis, stride=self.hop_length, padding=0,
        )
        cutoff = int(self.filter_length / 2 + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        return magnitude


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length, hop_length, win_length, n_mel_channels,
                 sampling_rate, mel_fmin, mel_fmax):
        super().__init__()
        self.stft_fn = AudioLDM_STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=filter_length,
                                    n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

    def mel_spectrogram(self, y):
        magnitudes = self.stft_fn.transform(y)
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = torch.log(torch.clamp(mel_output, min=1e-5))
        return mel_output


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

        # Normalize waveform the same way AudioLDM does (zero-mean, peak 0.5)
        audio_array = audio_array - np.mean(audio_array)
        audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-8)
        audio_array = 0.5 * audio_array

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

    # VAE expects raw log-mel (not mean/std-normalized values).
    log_mel = mel_stft.mel_spectrogram(chunk_waveform)  # (1, 64, T)

    no_of_frames = log_mel.shape[2]
    if no_of_frames >=1024:
        log_mel = log_mel[:, :, :1024]
    else:
        required_padding = 1024 - no_of_frames
        log_mel = torch.nn.functional.pad(log_mel, (0, required_padding))

    # Diffusers AudioLDM pipeline uses (B, 1, T, F) = (B, 1, 1024, 64).
    log_mel = log_mel.transpose(1, 2)  # (1, 1024, 64)
    return log_mel.unsqueeze(0)        # (1, 1, 1024, 64)

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

    mel_stft = TacotronSTFT(
        filter_length=1024, hop_length=160, win_length=1024,
        n_mel_channels=64, sampling_rate=16000, mel_fmin=0, mel_fmax=8000,
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
                    end_index = start_index + 160000
            
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
