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
import librosa,os
import random
import scipy
import pdb 
from itertools import repeat
import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

from params import params

random.seed(23)

def make_spectrum(filename=None, y=None, is_slice=False, feature_type='logmag', mode=None, FRAMELENGTH=400, SHIFT=160, _max=None, _min=None):
    if y is not None:
        y = y
    else:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype !='float32':
            y = np.float32(y)

    ### Normalize waveform
    y = y / np.max(abs(y)) # / 2.

    D = librosa.stft(y, n_fft=FRAMELENGTH, hop_length=SHIFT,win_length=FRAMELENGTH,window=scipy.signal.hamming)
    utt_len = D.shape[-1]
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    ### Feature type
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    else:
        Sxx = D

    if mode == 'mean_std':
        mean = np.mean(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))
        std = np.std(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))+1e-12
        Sxx = (Sxx-mean)/std  
    elif mode == 'minmax':
        Sxx = 2 * (Sxx - _min)/(_max - _min) - 1

    return Sxx, phase, len(y)


def transform(filename,indir,outdir):
  audio, sr = T.load_wav(filename)
  if params.sample_rate != sr:
    raise ValueError(f'Invalid sample rate {sr}.')
  audio = torch.clamp(audio[0] / 32767.5, -1.0, 1.0)

  mel_args = {
      'sample_rate': sr,
      'win_length': params.hop_samples * 4,
      'hop_length': params.hop_samples,
      'n_fft': params.n_fft,
      'f_min': 20.0,
      'f_max': sr / 2.0,
      'n_mels': params.n_mels,
      'power': 1.0,
      'normalized': True,
  }
  mel_spec_transform = TT.MelSpectrogram(**mel_args)

  with torch.no_grad():
    spectrogram = mel_spec_transform(audio)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    # print(spectrogram.shape)
    np.save(f'{filename.replace(indir,outdir)}.spec.npy', spectrogram.cpu().numpy()) 

def spec_transform(filename,outdir,spec_scp):
    spec, _, _ = make_spectrum(filename,FRAMELENGTH=params.n_fft, SHIFT=params.hop_samples)
    # outname = filename.replace(indir,outdir)+".spec.npy"
    outname = os.path.join(outdir, filename.split("/")[-2],filename.split("/")[-1]+".spec.npy")
    if not os.path.exists(os.path.join(outdir, filename.split("/")[-2])):
        os.makedirs(os.path.join(outdir, filename.split("/")[-2]))
    np.save(outname, spec)
    write_scp(outname, spec_scp)



# def choose_channel(n_files):
#     fins = []
#     n_ch_files = []
#     for n_ in n_files:
#         if n_.split(".")[0] in fins:
#             continue
#         n_ch_files.append(n_.split(".")[0] + ".CH" + str(random.randint(1,6))+ ".wav")
#         fins.append(n_.split(".")[0])
#     return n_ch_files


def read_scp(wav_scp, wav_dir):
    wav_file_list = [] 
    with open(wav_scp,"r") as scpfile:
        for line in scpfile.readlines():
            wav_file_list.append(os.path.join(wav_dir,line.split()[1]))
    return wav_file_list

def write_scp(outname, spec_scp):
    with open(spec_scp,"a") as scpfile:
        scpfile.write(f'{outname.split("/")[-1].split(".")[0]} {outname}\n')


def main(args):
  if args.se:
    params.n_mels = 513
  else:
    params.n_mels = 80

  filenames = read_scp(args.wav_scp, args.wav_dir)

  if args.se:
    with ProcessPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(spec_transform, filenames, repeat(args.outdir), repeat(args.spec_scp)), desc='Preprocessing', total=len(filenames)))
  else:
    with ProcessPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(transform, filenames,  repeat(args.outdir), repeat(args.spec_scp)), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train DiffWave')
  parser.add_argument('wav_scp', 
      help='input scp')
  parser.add_argument('wav_dir', 
      help='input scp dir')
  parser.add_argument('spec_scp', 
      help='output scp')
  parser.add_argument('outdir',
      help='output directory containing .npy files for training')
  parser.add_argument('--se', dest='se', action='store_true')
  parser.add_argument('--vocoder', dest='se', action='store_false')
  parser.add_argument('--train', dest='test', action='store_false')
  parser.add_argument('--test', dest='test', action='store_true')
  parser.set_defaults(se=True)
  parser.set_defaults(test=False)
  main(parser.parse_args())