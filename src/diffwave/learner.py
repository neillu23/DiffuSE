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
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb

from dataset import from_path as dataset_from_path
from model import DiffWave
from params import AttrDict


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


class DiffWaveLearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True

    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.L1Loss()
    self.summary_writer = None

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict, pretrain=False):
    if pretrain:
      print("WARNING: Remove parameters from model")
      model_state_dict = state_dict['model']
      for i in range(30):
        model_state_dict.pop("residual_layers.{}.conditioner_projection.weight".format(i), None)
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      if pretrain:
        # Missing key(s) in state_dict:"residual_layers.{i}.conditioner_projection.weight" 
        # Missing key(s) in state_dict:"residual_layers.{i}.output_residual.{weight,bias}" 
        self.model.module.load_state_dict(model_state_dict, strict=False)
      else:
        self.model.module.load_state_dict(state_dict['model'])
    else:
      if pretrain:
        self.model.load_state_dict(model_state_dict, strict=False)
      else:
        self.model.load_state_dict(state_dict['model'])
    
    if not pretrain:
      self.optimizer.load_state_dict(state_dict['optimizer'])
      self.scaler.load_state_dict(state_dict['scaler'])
      self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def restore_from_checkpoint(self,pretrain_path=None, filename='weights'):
    # pdb.set_trace()
    if pretrain_path!=None:
      print(f'load pretrain model at {pretrain_path}')
      checkpoint = torch.load(pretrain_path)
      self.load_state_dict(checkpoint,pretrain=True)
    else:
      try:
        checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
        self.load_state_dict(checkpoint)
        return True
      except FileNotFoundError:
        return False

  def fix_parameter2(self):
    self.fix_parameter()
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      for layer in self.model.module.residual_layers:
        for param in layer.output_projection.parameters():
          param.requires_grad = False
    else:
      for layer in self.model.residual_layers:
        for param in layer.output_projection.parameters():
          param.requires_grad = False


        
  def fix_parameter(self):
    # pdb.set_trace()
    for param in self.model.parameters():
      param.requires_grad = True
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      for param in self.model.module.skip_projection.parameters():
        param.requires_grad = False

      for param in self.model.module.output_projection.parameters():
        param.requires_grad = False
        
    else:
      for param in self.model.skip_projection.parameters():
        param.requires_grad = False

      for param in self.model.output_projection.parameters():
        param.requires_grad = False

  def train(self, max_steps=None):
    device = next(self.model.parameters()).device
    while True:
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss = self.train_step(features)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % 50 == 0:
            self._write_summary(self.step, features, loss)
          if self.step % len(self.dataset) == 0:
            self.save_to_checkpoint()
        self.step += 1

  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    audio = features['audio']
    spectrogram = features['spectrogram']

    N, T = audio.shape
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
      t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
      noise_scale = self.noise_level[t].unsqueeze(1)
      noise_scale_sqrt = noise_scale**0.5
      noise = torch.randn_like(audio)
      noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise

      predicted = self.model(noisy_audio, spectrogram, t)
      loss = self.loss_fn(noise, predicted.squeeze(1))

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss

  def _write_summary(self, step, features, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)
    writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer


def _train_impl(replica_id, model, dataset, args, params):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = DiffWaveLearner(args.model_dir, model, dataset, opt, params, fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  learner.restore_from_checkpoint(args.pretrain_path)
  if args.pretrain_path != None and args.fix:
    learner.fix_parameter()
  if args.pretrain_path != None and args.fix2:
    learner.fix_parameter2()
  learner.train(max_steps=args.max_steps)


def train(args, params):
  dataset = dataset_from_path(args.clean_dir, args.data_dirs, params, se=args.se)
  model = DiffWave(args, params).cuda()
  _train_impl(0, model, dataset, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)

  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  model = DiffWave(args, params).to(device)
  model = DistributedDataParallel(model, device_ids=[replica_id], find_unused_parameters=True)
  _train_impl(replica_id, model, dataset_from_path(args.clean_dir, args.data_dirs, params, se=args.se, is_distributed=True), args, params)
