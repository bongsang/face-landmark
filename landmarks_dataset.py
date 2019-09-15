"""
    - Author: Bongsang Kim
    - homepage: https://bongsang.github.io
    - Linkedin: https://www.linkedin.com/in/bongsang
"""
import os
from skimage import io, transform

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class LandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)

        # if image has an alpha color channel, get rid of it
        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}
        print(f'LandmarksDataset shape, image={image.shape}, landmarks={landmarks.shape}')

        if self.transform:
            sample = self.transform(sample)

        return sample
