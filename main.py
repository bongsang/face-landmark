"""
    - Author: Bongsang Kim
    - homepage: https://bongsang.github.io
    - Linkedin: https://www.linkedin.com/in/bongsang
"""

import glob
import os
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import skimage.io as io
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from landmarks_dataset import LandmarksDataset
from landmarks_transform import Rescale, RandomCrop, Normalize, ToTensor


def get_image_name(frame, idx):
    return frame.iloc[idx, 0]


def get_landmarks(frame, idx):
    landmark_points = frame.iloc[idx, 1:].to_numpy()
    landmark_points = landmark_points.astype('float').reshape(-1, 2)

    return landmark_points


def show_landmarks(image, points):
    plt.imshow(image)
    plt.scatter(points[:, 0], points[:, 1], s=20, marker='.', c='m')
    plt.pause(0.001)


# select an image by index in our data frame
def demo_show_landmarks_dataframe(landmarks_frame, idx):
    # landmarks_frame = pd.read_csv('data/training_frames_keypoints.csv')
    image_path = os.path.join('data/training/', get_image_name(landmarks_frame, idx))
    image_data = io.imread(image_path)
    print(image_data.shape)
    # print(image_path)
    landmarks = get_landmarks(landmarks_frame, idx)
    show_landmarks(image_data, landmarks)


def demo_show_landmarks_dataset(landmarks_dataset, idx):
    # face_dataset = LandmarksDataset(csv_file='data/training_frames_keypoints.csv',
    #                                 root_dir='data/training/')
    # print('Length of torch dataset: ', len(landmarks_dataset))
    # print(landmarks_dataset[0]['image'].shape)

    sample = landmarks_dataset[idx]
    show_landmarks(sample['image'], sample['landmarks'])


def demo_show_transformation(landmarks_dataset):
    # test out some of these transforms
    rescale = Rescale(100)
    crop = RandomCrop(50)
    composed = transforms.Compose([Rescale(250),
                                   RandomCrop(224)])

    # apply the transforms to a sample image
    test_num = 0
    sample = landmarks_dataset[test_num]

    fig = plt.figure()
    for i, tx in enumerate([rescale, crop, composed]):
        transformed_sample = tx(sample)

        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tx).__name__)
        show_landmarks(transformed_sample['image'], transformed_sample['landmarks'])


def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')


if __name__ == '__main__':
    '''Pandas landmark data frame
    '''
    landmarks_frame = pd.read_csv('data/training_frames_keypoints.csv')
    print(landmarks_frame.describe())
    print(landmarks_frame.head())

    landmarks_dataset = LandmarksDataset(csv_file='data/training_frames_keypoints.csv',
                                         root_dir='data/training/')
    # demo_show_landmarks_dataframe(landmarks_frame, np.random.randint(0, landmarks_frame.shape[0]))
    # plt.show()

    '''PyTorch landmark data set
    '''
    # demo_show_landmarks_dataset(landmarks_dataset, np.random.randint(0, len(landmarks_dataset)))
    # demo_show_transformation(landmarks_dataset)
    # plt.show()

    ''' PyTorch Dataset Transform to tensor
    '''
    data_transform = transforms.Compose([Rescale(256),
                                         RandomCrop(224),
                                         # Normalize(),
                                         ToTensor()])
    transformed_landmarks_dataset = LandmarksDataset(csv_file='data/training_frames_keypoints.csv',
                                                     root_dir='data/training/',
                                                     transform=data_transform)
    for i in range(5):
        sample = transformed_landmarks_dataset[i]
        # numpy image: H x W x C
        # torch image: C X H X W
        print(i, sample['image'].size(), sample['landmarks'].size())

    ''' PyTorch Dataset Batch
    '''
    data_loader = DataLoader(transformed_landmarks_dataset, batch_size=4,
                             shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(data_loader):
        print(i_batch, sample_batched['image'].size(), sample_batched['landmarks'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
