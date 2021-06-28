# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import random
import torch
import torchaudio

from glob import glob
from torch.utils.data.distributed import DistributedSampler


class NumpyDataset(torch.utils.data.Dataset):
  def __init__(self, wav_path, npy_paths, se, voicebank=False):
    super().__init__()
    # self.filenames = []
    self.wav_path = wav_path
    self.se = se
    self.voicebank = voicebank
    print(npy_paths,wav_path)

    specnames = []
    for path in npy_paths:
      specnames += glob(f'{path}/*.wav.spec.npy', recursive=True)
    
    self.signals = []
    self.spectrograms = []
    for spec_filename in specnames:
      spec_path = "/".join(spec_filename.split("/")[:-1])
      audio_filename = spec_filename.replace(spec_path, self.wav_path).replace(".spec.npy", "")
      signal, _ = torchaudio.load_wav(audio_filename)
      spectrogram = np.load(spec_filename)
      self.signals.append(signal)
      self.spectrograms.append(spectrogram)





  def __len__(self):
    return len(self.spectrograms)

  def __getitem__(self, idx):
    signal = self.signals[idx]
    spectrogram = self.spectrograms[idx]
    return {
        'audio': signal[0] / 32767.5,
        'spectrogram': spectrogram.T
    }


class Collator:
  def __init__(self, params):
    self.params = params

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    for record in minibatch:
      # Filter out records that aren't long enough.
      if len(record['spectrogram']) < self.params.crop_mel_frames:
        del record['spectrogram']
        del record['audio']
        continue

      start = random.randint(0, record['spectrogram'].shape[0] - self.params.crop_mel_frames)
      end = start + self.params.crop_mel_frames
      record['spectrogram'] = record['spectrogram'][start:end].T

      start *= samples_per_frame
      end *= samples_per_frame
      record['audio'] = record['audio'][start:end]
      record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')

    audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
    spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
    return {
        'audio': torch.from_numpy(audio),
        'spectrogram': torch.from_numpy(spectrogram),
    }


def from_path(clean_dir, data_dirs, params, se=True, voicebank=False, is_distributed=False):
  dataset = NumpyDataset(clean_dir, data_dirs, se, voicebank)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate,
      shuffle=not is_distributed,
      num_workers=os.cpu_count(),
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)
