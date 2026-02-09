# SPDX-License-Identifier: MIT
# Copyright (c) 2026-present
"""
Dataset loading and preprocessing for SNN models.

This code is modified from:
https://github.com/Thvnvtos/SNN-delays
"""

from utils import set_seed

import numpy as np

from torch.utils.data import DataLoader
from typing import Callable, Optional

import torchvision.transforms as transforms

from spikingjelly.datasets.shd import SpikingHeidelbergDigits
from spikingjelly.datasets.shd import SpikingSpeechCommands
from spikingjelly.datasets import pad_sequence_collate

import torch
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB, Resample
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from torchvision import transforms
from torch.utils.data import Dataset
import augmentations


class RNoise(object):
  
  def __init__(self, sig):
    self.sig = sig
        
  def __call__(self, sample):
    noise = np.abs(np.random.normal(0, self.sig, size=sample.shape).round())
    return sample + noise


class TimeNeurons_mask_aug(object):

  def __init__(self, config):
    self.config = config
  
  
  def __call__(self, x, y):
    # Sample shape: (time, neurons)
    for sample in x:
      # Time mask
      if np.random.uniform() < self.config.TN_mask_aug_proba:
        mask_size = np.random.randint(0, self.config.time_mask_size)
        ind = np.random.randint(0, sample.shape[0] - self.config.time_mask_size)
        sample[ind:ind+mask_size, :] = 0

      # Neuron mask
      if np.random.uniform() < self.config.TN_mask_aug_proba:
        mask_size = np.random.randint(0, self.config.neuron_mask_size)
        ind = np.random.randint(0, sample.shape[1] - self.config.neuron_mask_size)
        sample[:, ind:ind+mask_size] = 0

    return x, y


class CutMix(object):
  """
  Apply Spectrogram-CutMix augmentaiton which only cuts patch across time axis unlike 
  typical Computer-Vision CutMix. Applies CutMix to one batch and its shifted version.
    
  """

  def __init__(self, config):
    self.config = config
  
  
  def __call__(self, x, y):
    
    # x shape: (batch, time, neurons)
    # Go to L-1, no need to augment last sample in batch (for ease of coding)

    for i in range(x.shape[0]-1):
      # other sample to cut from
      j = i+1
      
      if np.random.uniform() < self.config.cutmix_aug_proba:
        lam = np.random.uniform()
        cut_size = int(lam * x[j].shape[0])

        ind = np.random.randint(0, x[i].shape[0] - cut_size)

        x[i][ind:ind+cut_size, :] = x[j][ind:ind+cut_size, :]

        y[i] = (1-lam) * y[i] + lam * y[j]

    return x, y


class Augs(object):

  def __init__(self, config):
    self.config = config
    self.augs = [TimeNeurons_mask_aug(config), CutMix(config)]
  
  def __call__(self, x, y):
    for aug in self.augs:
      x, y = aug(x, y)
    
    return x, y


def SHD_dataloaders(config):
  set_seed(config.seed)

  train_dataset = BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=True, data_type='frame', duration=config.time_step)
  test_dataset= BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=False, data_type='frame', duration=config.time_step)

  #train_dataset, valid_dataset = random_split(train_dataset, [0.8, 0.2])

  train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True, num_workers=4)
  #valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size)
  test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

  return train_loader, test_loader

class SpikeDataset(Dataset):
    def __init__(self, X, Y, dataset_name="unknown"):
        self.X = X
        self.Y = Y
        self.dataset_name = dataset_name
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        x = self.X[idx].T
        y = self.Y[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long), torch.tensor(x.shape[-1])


# === Create DataLoaders ===
def create_dataloaders(X, Y, dataset_name, batch_size=128, train_ratio=0.6, val_ratio=0.15, test_ratio=0.15):
    """
    Create train, validation, and test DataLoaders from dataset.
    
    Args:
        X: Input spike data
        Y: Labels
        dataset_name: Name of the dataset for logging
        batch_size: Batch size for DataLoaders
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if X is None or Y is None:
        return None, None, None
    
    N = len(Y)
    
    # Define splits
    train_end = int(N * train_ratio)
    val_end = int(N * (train_ratio + val_ratio))
    test_end = int(N * (train_ratio + val_ratio + test_ratio))
    
    # Split indices
    train_indices = np.arange(0, train_end)
    val_indices = np.arange(train_end, val_end)
    test_indices = np.arange(val_end, test_end)
    
    # Shuffle training indices
    np.random.shuffle(train_indices)
    
    # Create datasets
    train_dataset = SpikeDataset(X[train_indices], Y[train_indices], f"{dataset_name}_train")
    val_dataset = SpikeDataset(X[val_indices], Y[val_indices], f"{dataset_name}_val")
    test_dataset = SpikeDataset(X[test_indices], Y[test_indices], f"{dataset_name}_test")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"     {dataset_name} DataLoaders created:")
    print(f"      Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"      Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"      Test:  {len(test_dataset)} samples ({len(test_loader)} batches)")
    
    return train_loader, val_loader, test_loader


def SHD_NORM_dataloaders(config):

    # Load SHD-Norm
    X_shd, Y_shd = load_shd_norm()
    shd_train_loader, shd_val_loader, shd_test_loader = create_dataloaders(
    X_shd, Y_shd, "SHD-Norm", batch_size=config.batch_size)

    return shd_train_loader, shd_val_loader, shd_test_loader

def SSC_NORM_dataloaders(config):

    # Load SSC-Norm  
    X_ssc, Y_ssc = load_ssc_norm()
    ssc_train_loader, ssc_val_loader, ssc_test_loader = create_dataloaders(
        X_ssc, Y_ssc, "SSC-Norm", batch_size=config.batch_size)
  
    return ssc_train_loader, ssc_val_loader, ssc_test_loader


def SSC_dataloaders(config):
  set_seed(config.seed)


  if getattr(config, 'data_from_dcls_code_base', True):

    train_dataset = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split='train', data_type='frame', duration=config.time_step)
    valid_dataset = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split='valid', data_type='frame', duration=config.time_step)
    test_dataset = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split='test', data_type='frame', duration=config.time_step)


    train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

  else:
    import sys
    sys.path.append('/Users/ybouhadjar/projects/SSM-inspired-LIF/dataloaders')
    from spiking_datasets import load_shd_or_ssc

    train_loader = load_shd_or_ssc(
                    dataset_name='ssc',
                    data_folder=config.datasets_path,
                    split="train",
                    batch_size=config.batch_size,
                    nb_steps=config.nb_steps,
                    max_time = config.max_time,
                    spatial_bin = config.n_bins,
                    shuffle=True,
                    workers=4,
                    dcls_code_base=True
                )
    valid_loader = load_shd_or_ssc(
                    dataset_name='ssc',
                    data_folder=config.datasets_path,
                    split="valid",
                    batch_size=config.batch_size,
                    nb_steps=config.nb_steps,
                    max_time = config.max_time,
                    spatial_bin = config.n_bins,
                    shuffle=False,
                    workers=4,
                    dcls_code_base=True
                )
    test_loader = load_shd_or_ssc(
                        dataset_name='ssc',
                        data_folder=config.datasets_path,
                        split="test",
                        batch_size=config.batch_size,
                        nb_steps=config.nb_steps,
                        max_time = config.max_time,
                        spatial_bin = config.n_bins,
                        shuffle=False,
                        workers=4,
                        dcls_code_base=True
                    )

  return train_loader, valid_loader, test_loader

def GSC_dataloaders(config):
  set_seed(config.seed)

  if config.data_from_dcls_code_base:

    train_dataset = GSpeechCommands(config.datasets_path, 'training', transform=build_transform(False), target_transform=target_transform)
    valid_dataset = GSpeechCommands(config.datasets_path, 'validation', transform=build_transform(False), target_transform=target_transform)
    test_dataset = GSpeechCommands(config.datasets_path, 'testing', transform=build_transform(False), target_transform=target_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)
  
  else:

    import sys
    sys.path.append('/Users/ybouhadjar/projects/SSM-inspired-LIF/dataloaders')
    from nonspiking_datasets import load_hd_or_sc

    train_loader = load_hd_or_sc(
                    dataset_name='sc',
                    data_folder=config.datasets_path,
                    split="train",
                    batch_size=config.batch_size,
                    use_augm=False,
                    shuffle=True,
                    workers=4,
                    dcls_code_base=True
                )
    valid_loader = load_hd_or_sc(
                    dataset_name='sc',
                    data_folder=config.datasets_path,
                    split="valid",
                    batch_size=config.batch_size,
                    use_augm=False,
                    shuffle=False,
                    workers=4,
                    dcls_code_base=True
                )
    test_loader = load_hd_or_sc(
                        dataset_name='sc',
                        data_folder=config.datasets_path,
                        split="test",
                        batch_size=config.batch_size,
                        use_augm=False,
                        shuffle=False,
                        workers=4,
                        dcls_code_base=True
                    )

  return train_loader, valid_loader, test_loader


class BinnedSpikingHeidelbergDigits(SpikingHeidelbergDigits):
    def __init__(
            self,
            root: str,
            n_bins: int,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """
        The Spiking Heidelberg Digits (SHD) dataset, which is proposed by `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.

        .. admonition:: Note
            :class: note

            Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``. Thus, this dataset is not inherited from :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` directly. But their procedures are similar.

        :class:`spikingjelly.datasets.shd.custom_integrate_function_example` is an example of ``custom_integrate_function``, which is similar to the cunstom function for DVS Gesture in the ``Neuromorphic Datasets Processing`` tutorial.
        """
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
        self.n_bins = n_bins

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
            label = self.h5_file['labels'][i]
            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return events, label

        elif self.data_type == 'frame':
            frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
            label = self.frames_label[i]

            binned_len = frames.shape[1]//self.n_bins
            binned_frames = np.zeros((frames.shape[0], binned_len))
            for i in range(binned_len):
                binned_frames[:,i] = frames[:, self.n_bins*i : self.n_bins*(i+1)].sum(axis=1)

            if self.transform is not None:
                binned_frames = self.transform(binned_frames)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return binned_frames, label



class BinnedSpikingSpeechCommands(SpikingSpeechCommands):
    def __init__(
            self,
            root: str,
            n_bins: int,
            split: str = 'train',
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """
        The Spiking Speech Commands (SSC) dataset, which is proposed by `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.

        .. admonition:: Note
            :class: note

            Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``. Thus, this dataset is not inherited from :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` directly. But their procedures are similar.

        :class:`spikingjelly.datasets.shd.custom_integrate_function_example` is an example of ``custom_integrate_function``, which is similar to the cunstom function for DVS Gesture in the ``Neuromorphic Datasets Processing`` tutorial.
        """
        super().__init__(root, split, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
        self.n_bins = n_bins

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
            label = self.h5_file['labels'][i]
            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return events, label

        elif self.data_type == 'frame':
            frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
            label = self.frames_label[i]

            binned_len = frames.shape[1]//self.n_bins
            binned_frames = np.zeros((frames.shape[0], binned_len))
            for i in range(binned_len):
                binned_frames[:,i] = frames[:, self.n_bins*i : self.n_bins*(i+1)].sum(axis=1)

            if self.transform is not None:
                binned_frames = self.transform(binned_frames)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return binned_frames, label


def build_transform(is_train):
    sample_rate=16000
    window_size=256
    hop_length=80
    n_mels=140
    f_min=50
    f_max=14000

    t = [augmentations.PadOrTruncate(sample_rate),
         Resample(sample_rate, sample_rate // 2)]
    if is_train:
        t.extend([augmentations.RandomRoll(dims=(1,)),
                  augmentations.SpeedPerturbation(rates=(0.5, 1.5), p=0.5)
                 ])

    t.append(Spectrogram(n_fft=window_size, hop_length=hop_length, power=2))

    if is_train:
        pass

    t.extend([MelScale(n_mels=n_mels,
                       sample_rate=sample_rate // 2,
                       f_min=f_min,
                       f_max=f_max,
                       n_stft=window_size // 2 + 1),
              AmplitudeToDB()
             ])

    return transforms.Compose(t)

labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

target_transform = lambda word : torch.tensor(labels.index(word))

class GSpeechCommands(Dataset):
    def __init__(self, root, split_name, transform=None, target_transform=None, download=True):

        self.split_name = split_name
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = SPEECHCOMMANDS(root, download=download, subset=split_name)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        waveform, _,label,_,_ = self.dataset.__getitem__(index)

        if self.transform is not None:
            waveform = self.transform(waveform).squeeze().t()

        target = label

        if self.target_transform is not None:
            target = self.target_transform(target)

        return waveform, target, torch.zeros(1)
