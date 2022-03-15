import h5py
from pathlib import Path
from typing import List, Union
import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio

ORIGINAL_SAMPLE_RATE = 12000

class Dataset(Dataset):
    
    def __init__(
        self,
        train: bool,
        data_info: dict,
        data_path: str,
        direction: Union[str, List[str]],
        target_sample_rate: int = 1000,
        n_fft: int = 256,
        hop_length: int = 128,
        n_columns: int = 32,
        hop_columns: int = 4,
    ) -> None:

        self.train = train
        self.data_info = data_info
        self.data_path = data_path
        self.direction = [direction] if isinstance(direction, str) else direction
        self.direction = [d.upper() for d in self.direction]
        self.n_directions = len(self.direction)

        self.target_sample_rate = target_sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_columns = n_columns
        self.hop_columns = hop_columns

        self.transformation = \
            torchaudio.transforms.Spectrogram(n_fft = n_fft, hop_length = hop_length)
        self.power_to_db = torchaudio.transforms.AmplitudeToDB()
        
        self._load_data()
        self._normalize_data()

    #======Dataset methods========================================================= 
    def __len__(self) -> int:
        return len(self.info)

    def __getitem__(self, idx: int) -> Tensor:
        spec, pos_ini, scenario = self.info[idx].values()
        spec = self.specs[spec][:,:,pos_ini:pos_ini+self.n_columns]
        return spec, scenario

    #======Raw Audio=========================================================
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _normalize_raw(self, signal):
        return (signal-torch.mean(signal))/torch.std(signal)

    #======Spectrogram=========================================================
    def _read_csv(self, path):
        data = pd.read_csv(path, header=3, encoding='ISO-8859-1', engine='python')
        data = data.iloc[:,1].to_numpy()
        return torch.Tensor(data[~np.isnan(data)]) 

    def _log_spectrogram(self, signal):
        signal = self._resample_if_necessary(signal, ORIGINAL_SAMPLE_RATE)
        signal = self._normalize_raw(signal)
        signal = self.transformation(signal)
        signal = self.power_to_db(signal)
        signal = signal[:self.n_fft//2,:]
        return signal

    def _load_data(self):
        self.specs, self.info = [], []
        for scenario in self.data_info:
            dates = self.data_info[scenario]
            for date in dates:
                for audio in dates[date]:
                    spec = []
                    for direction in self.direction:
                        path = "{}/{}/{}/waveform.AFILADOR_{}_HF_{}.csv".format(
                            self.data_path, date, scenario, direction, audio)
                        spec.append(self._log_spectrogram(self._read_csv(path)))
                    spec = torch.stack(spec)
                    self.info.extend([{'spec':len(self.specs), 'pos_ini':pos_ini, 'scenario': scenario,} \
                        for pos_ini in range(0, spec.shape[1]-self.n_columns, self.hop_columns)])
                    self.specs.append(spec)

    def _normalize_data(self):
        Path(Path(__file__).resolve().parents[0]/"stats").mkdir(parents=True, exist_ok=True)
        stats_spec_path = self.stats_spec_path = \
            "{path}/stats/{direction}_sr_{sr}_N_{N}_M_{M}.h5".format(
                path=Path(__file__).resolve().parents[0], 
                direction='_'.join(self.direction),
                sr=self.target_sample_rate, 
                N=self.n_fft, 
                M=self.hop_length
            )

        if self.train:
            mean = torch.stack(self.specs).mean(dim=(0,2,3))
            std = torch.stack(self.specs).std(dim=(0,2,3))
            with h5py.File(stats_spec_path, 'w') as f:
                f.create_dataset('mean', data=mean)
                f.create_dataset('std', data=std)
        else:
            with h5py.File(stats_spec_path, 'r') as f:
                mean, std = f['mean'][()], f['std'][()]
            
        for i in range(len(self.specs)):
            self.specs[i] = ((self.specs[i].permute(2,1,0)-mean)/std).permute(2,1,0)