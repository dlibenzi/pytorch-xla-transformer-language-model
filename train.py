# Copyright (c) 2019, Bryan McCann
# All rights reserved.

import os
import time
import math

import numpy
import torch
import torch.utils.data

import torch_xla
import torch_xla_py.xla_model as xm
import torch_xla_py.data_parallel as dp

from transformer import Transformer


class LazyDataset:

  def __init__(self, path, sequence_length):
    self.path = path
    self.size = os.stat(path).st_size - sequence_length
    self.sequence_length = sequence_length

  def __getitem__(self, index):
    with open(self.path, 'rb') as f:
      f.seek(index)
      chunk = f.read(self.sequence_length)
    return torch.ByteTensor(numpy.frombuffer(chunk, dtype=numpy.uint8))

  def __len__(self):
    return self.size


def get_dataloader(path, batch_size, sequence_length, num_workers):
  dataset = LazyDataset(path, sequence_length + 1)
  sampler = torch.utils.data.RandomSampler(dataset)
  return torch.utils.data.DataLoader(
      dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)


def main():
  BATCH_SIZE = 128
  LOG_STEPS = 10
  METRICS_STEP = 50
  NUM_EPOCHS = 8
  SEQUENCE_LENGTH = 256

  devices = xm.get_xla_supported_devices()
  model = Transformer(256, 12, 512, 2048, 8, 0.2)
  model_parallel = dp.DataParallel(model, device_ids=devices)

  def train_loop_fn(model, loader, device, context):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    tracker = xm.RateTracker()

    positions = torch.arange(SEQUENCE_LENGTH).long().view(
        1, SEQUENCE_LENGTH).to(device)
    causal_mask = torch.triu(
        torch.ones(
            SEQUENCE_LENGTH, SEQUENCE_LENGTH, dtype=torch.uint8, device=device),
        diagonal=1).unsqueeze(0)

    for iteration, batch in loader:
      input = batch[:, :-1].long()
      target = batch[:, 1:].long()

      loss = model(input, positions, target, batch_mask=causal_mask)
      loss.backward()
      xm.optimizer_step(optimizer)

      tracker.add(BATCH_SIZE)
      if iteration % LOG_STEPS == 0:
        print('[{}]({}) Loss={:.5f} Rate={:.2f}'.format(
            device, iteration,
            loss.item() / math.log(2), tracker.rate()))
      if iteration % METRICS_STEP == 0:
        print(torch_xla._XLAC._xla_metrics_report())

  train_loader = get_dataloader('datasets/enwik8/train/train.txt.raw',
                                BATCH_SIZE, SEQUENCE_LENGTH, 0)

  for epoch in range(0, NUM_EPOCHS):
    model_parallel(train_loop_fn, train_loader)


if __name__ == '__main__':
  main()
