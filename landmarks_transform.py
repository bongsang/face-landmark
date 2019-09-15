"""
    - Author: Bongsang Kim
    - homepage: https://bongsang.github.io
    - Linkedin: https://www.linkedin.com/in/bongsang
"""

import numpy as np
import cv2

import torch
from torchvision import transforms, utils


class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        image_copy = np.copy(image)
        landmarks_copy = np.copy(landmarks)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # scale color range from [0, 255] to [0, 1]
        image_copy = image_copy / 255.0

        # scale landmarks to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        landmarks_copy = (landmarks_copy - 100) / 50.0

        return {'image': image_copy, 'landmarks': landmarks_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image_copy = cv2.resize(image, (new_w, new_h))
        landmarks_copy = landmarks * [new_w / w, new_h / h]

        return {'image': image_copy, 'landmarks': landmarks_copy}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image_copy = image[top: top + new_h, left: left + new_w]
        landmarks_copy = landmarks - [left, top]

        return {'image': image_copy, 'landmarks': landmarks_copy}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # if image has no grayscale color channel, add one
        if len(image.shape) == 2:
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        print(f'LandmarksDataset torch tensor shape, image={image.shape}, landmarks={landmarks.shape}')

        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}
