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
import pdb
import os
import torch
import torchaudio
import librosa
import random
from argparse import ArgumentParser

from params import AttrDict, params as base_params
from model import DiffWave

from os import path
from glob import glob
from tqdm import tqdm

random.seed(23)

models = {}


def load_model(model_dir=None, args=None, params=None, device=torch.device('cuda')):
  # Lazy load model.
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)
    model = DiffWave(args, AttrDict(base_params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_dir] = model
  model = models[model_dir]
  model.params.override(params)
      
  return model
      

def predict(spectrogram, model, noisy_wav,alpha, beta, alpha_cum, T, device=torch.device('cuda')):
  with torch.no_grad():
    
    # Expand rank 2 tensors by adding a batch dimension.
    if len(spectrogram.shape) == 2:
      spectrogram = spectrogram.unsqueeze(0)
    spectrogram = spectrogram.to(device)

    audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
    # audio = torch.randn(spectrogram.shape[0], wlen, device=device)
    noisy_audio = torch.from_numpy(noisy_wav)
    # audio[0,:noisy_audio.shape[0]] = noisy_audio
    # pdb.set_trace()
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

    for n in range(len(alpha) - 1, -1, -1):
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      audio = c1 * (audio - c2 * model(audio, spectrogram, torch.tensor([T[n]], device=audio.device)).squeeze(1))
      if n > 0:
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise
      audio = torch.clamp(audio, -1.0, 1.0)
  return audio, model.params.sample_rate

# def narrow(audio,wlen):
#     # if audio.shape[0]:
#     audio = audio[:wlen,:]
#     return audio

# def extend(audio,wlen):
#     if wlen - audio.shape[0] >  1024
    
#     return audio


def read_scp(wav_scp, wav_path, spec_scp):
    spec_file_list = [] 

    len_dir = {}
    wav_dir = {}
    with open(wav_scp,"r") as wavscp:
        for line in wavscp.readlines():
            name = line.split()[0]
            wavename = os.path.join(wav_path,line.split()[1])
            wav, _ = librosa.load(wavename,sr=16000)
            len_dir[name] = wav.shape[0]
            wav_dir[name] = wav

    with open(spec_scp,"r") as scpfile:
        for line in scpfile.readlines():
            name = line.split()[0]
            specname = line.split()[1]
            wave_len = len_dir[name]
            wav = wav_dir[name]
            spec_file_list.append((specname,wave_len,wav))
    return spec_file_list



def inference_schedule(model, fast_sampling=False):
    training_noise_schedule = np.array(model.params.noise_schedule)
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    # print("alpha_cum",talpha_cum)
    # print("gamma_cum",alpha_cum)

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)
    return alpha, beta, alpha_cum, T
      


def main(args):
  if args.se:
    base_params.n_mels = 513
  else:
    base_params.n_mels = 80
  specnames = read_scp(args.wav_scp, args.wav_path, args.spec_scp)
  model = load_model(model_dir=args.model_dir ,args=args)
  alpha, beta, alpha_cum, T = inference_schedule(model, fast_sampling=args.fast)
  for spec, wlen, wav in tqdm(specnames):
    spectrogram = torch.from_numpy(np.load(spec))
    audio, sr = predict(spectrogram, model, wav, alpha, beta, alpha_cum, T)
    output_path = os.path.join(args.output, spec.split("/")[-3], spec.split("/")[-2])
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    output_name = os.path.join(output_path, spec.split("/")[-1].replace("flac.spec.npy", "wav"))
    # pdb.set_trace()
    audio = audio[:,:wlen]
    torchaudio.save(output_name, audio.cpu(), sample_rate=sr)
    write_scp(output_name,args.out_scp)


def write_scp(outname, out_scp):
    with open(out_scp,"a") as scpfile:
        scpfile.write(f'{outname.split("/")[-1].split(".")[0]} {outname}\n')


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('wav_scp',
      help='input wav scp for the length of the audio')
  parser.add_argument('wav_path',
      help='input wav scp directory')
  parser.add_argument('spec_scp',
      help='input spec scp')
  parser.add_argument('out_scp',
      help='output wav scp')
  # parser.add_argument('spectrogram_path', nargs='+',
  #     help='space separated list of directories from spectrogram file generated by diffwave.preprocess')
  parser.add_argument('--output', '-o', default='output/',
      help='output path name')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')
  parser.add_argument('--fix', dest='fix', action='store_true')
  parser.add_argument('--fix_in', dest='fix2', action='store_true')
  parser.add_argument('--se', dest='se', action='store_true')
  parser.add_argument('--vocoder', dest='se', action='store_false')
  parser.set_defaults(se=True)
  parser.set_defaults(fix=False)
  parser.set_defaults(fix_in=False)
  parser.set_defaults(voicebank=False)
  main(parser.parse_args())
