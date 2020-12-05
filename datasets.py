# License: MIT
# Author: Karl Stelzner

import os
import sys

import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.random import random_integers
from PIL import Image
import h5py


def make_sprite(height=64, width=64, background="random", min_num_objs=1, max_num_objs=3, shapes=[0,1,2], min_size=20):
    num_sprites = random_integers(min_num_objs, max_num_objs)
    image = np.zeros((height, width, 3))
    if background == "random":
        background_color = random_integers(0, 127, (3,))
        image[:, :, 0] = float(background_color[0]) / 255
        image[:, :, 1] = float(background_color[1]) / 255
        image[:, :, 2] = float(background_color[2]) / 255

    for j in range(num_sprites):
        pos_y = random_integers(0, height - min_size)
        pos_x = random_integers(0, width - min_size)

        scale = random_integers(min_size, min(min_size + 4, height - pos_y, width - pos_x))

        cat = np.random.choice(shapes, 1)
        sprite = np.zeros((height, width, 3))
        mask = np.ones((height, width, 3))

        if cat == 0:  # draw circle
            center_x = pos_x + scale // 2.0
            center_y = pos_y + scale // 2.0
            for x in range(height):
                for y in range(width):
                    dist_center_sq = (x - center_x) ** 2 + (y - center_y) ** 2
                    if dist_center_sq < (scale // 2.0) ** 2:
                        sprite[x][y][cat] = 1.0
                        mask[x][y][:] = 0.
        elif cat == 1:  # draw square
            sprite[pos_x:pos_x + scale, pos_y:pos_y + scale, cat] = 1.0
            mask[pos_x:pos_x + scale, pos_y:pos_y + scale, :] = 0.
        else:  # draw square turned by 45 degrees
            center_x = pos_x + scale // (np.sqrt(2.))
            center_y = pos_y + scale // (np.sqrt(2.))
            for x in range(height):
                for y in range(width):
                    if abs(x - center_x) + abs(y - center_y) < (scale // (np.sqrt(2.))):
                        sprite[x][y][cat] = 1.0
                        mask[x][y][:] = 0.
        image = np.clip(np.multiply(image, mask) + sprite, 0.0, 1.0)

    return image, num_sprites


def make_sprites(n=50000, height=64, width=64, background="random", min_num_objs=1, max_num_objs=3):
    print(n)
    images = np.zeros((n, height, width, 3))
    counts = np.zeros((n,))
    min_size = 16
    print('Generating sprite dataset...')

    for i in range(n):
        image, count = make_sprite(height=height, width=width, background=background, min_num_objs=min_num_objs, min_size=min_size, max_num_objs=max_num_objs, shapes=[0,1,2])
        images[i] = image
        counts[i] = count

    print("Finished generating dataset.")

    return {'x_train': images[:19 * n // 20],
            'count_train': counts[:19 * n // 20],
            'x_test': images[19 * n // 20:],
            'count_test': counts[19 * n // 20:]}
# def make_sprites(n=50000, height=64, width=64, background="random", min_num_objs=1, max_num_objs=3):
#     print(n)
#     images = np.zeros((n, height, width, 3))
#     counts = np.zeros((n,))
#     min_size = 20
#     print('Generating sprite dataset...')
#
#     for i in range(n):
#         num_sprites = random_integers(min_num_objs, max_num_objs)
#         counts[i] = num_sprites
#         if background == "random":
#             background_color = random_integers(0, 127, (3,))
#             images[i, :, :, 0] = float(background_color[0]) / 255
#             images[i, :, :, 1] = float(background_color[1]) / 255
#             images[i, :, :, 2] = float(background_color[2]) / 255
#
#         for j in range(num_sprites):
#             pos_y = random_integers(0, height - min_size)
#             pos_x = random_integers(0, width - min_size)
#
#             scale = random_integers(min_size, min(min_size + 4, height - pos_y, width - pos_x))
#
#             cat = random_integers(0, 2)
#             sprite = np.zeros((height, width, 3))
#             mask = np.ones((height, width, 3))
#
#             if cat == 0:  # draw circle
#                 center_x = pos_x + scale // 2.0
#                 center_y = pos_y + scale // 2.0
#                 for x in range(height):
#                     for y in range(width):
#                         dist_center_sq = (x - center_x) ** 2 + (y - center_y) ** 2
#                         if dist_center_sq < (scale // 2.0) ** 2:
#                             sprite[x][y][cat] = 1.0
#                             mask[x][y][:] = 0.
#             elif cat == 1:  # draw square
#                 sprite[pos_x:pos_x + scale, pos_y:pos_y + scale, cat] = 1.0
#                 mask[pos_x:pos_x + scale, pos_y:pos_y + scale, :] = 0.
#             else:  # draw square turned by 45 degrees
#                 center_x = pos_x + scale // (np.sqrt(2.))
#                 center_y = pos_y + scale // (np.sqrt(2.))
#                 for x in range(height):
#                     for y in range(width):
#                         if abs(x - center_x) + abs(y - center_y) < (scale // (np.sqrt(2.))):
#                             sprite[x][y][cat] = 1.0
#                             mask[x][y][:] = 0.
#             images[i] = np.multiply(images[i], mask) + sprite
#
#     images = np.clip(images, 0.0, 1.0)
#     print(19 * n // 20)
#
#     return {'x_train': images[:19 * n // 20],
#             'count_train': counts[:19 * n // 20],
#             'x_test': images[19 * n // 20:],
#             'count_test': counts[19 * n // 20:]}
#

class Sprites(Dataset):
    def __init__(self, directory, n=40000, rnd_background=False, canvas_size=64,
                 train=True, transform=None, min_num_objs=1, max_num_objs=3):
        np_file = 'sprites_{}_{}_min{}_max{}{}.hdf5'.format(n, canvas_size,min_num_objs, max_num_objs, "_rndbkg" if rnd_background else "")
        full_path = os.path.join(directory, np_file)
        if not os.path.isfile(full_path):
            gen_data = make_sprites(n, canvas_size, canvas_size,
                                    background="random" if rnd_background else None, min_num_objs=min_num_objs, max_num_objs=max_num_objs)
            with h5py.File(full_path, "w") as f:
                data = gen_data["x_train"].astype(float)
                dset = f.create_dataset("x_train", data.shape,
                                        dtype='f',
                                        data=data,
                                        compression="gzip",
                                        compression_opts=9)

                data = gen_data["x_test"].astype(float)
                dset = f.create_dataset("x_test", data.shape,
                                        dtype='f',
                                        data=data,
                                        compression="gzip",
                                        compression_opts=9)

                data = gen_data["count_train"].astype(float)
                dset = f.create_dataset("count_train", data.shape,
                                        dtype='f',
                                        data=data,
                                        compression="gzip",
                                        compression_opts=9)

                data = gen_data["count_test"].astype(float)
                dset = f.create_dataset("count_test", data.shape,
                                        dtype='f',
                                        data=data,
                                        compression="gzip",
                                        compression_opts=9)

        data = h5py.File(full_path, "r")

        self.transform = transform
        self.images = data['x_train'][:] if train else data['x_test'][:]
        self.counts = data['count_train'][:] if train else data['count_test'][:]
        print(self.images.shape)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.counts[idx]


class Coinrun(Dataset):
    def __init__(self, directory, dataset_name, train=True, transform=None ):
        full_path = os.path.join(directory, dataset_name)

        data = h5py.File(full_path, "r")

        self.transform = transform
        self.images = data['x_train'][:] if train else data['x_test'][:]
        print(self.images.shape)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.empty([])


class Clevr(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.filenames = os.listdir(directory)
        self.n = len(self.filenames)
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        imgpath = os.path.join(self.directory, self.filenames[idx])
        img = Image.open(imgpath)
        if self.transform is not None:
            img = self.transform(img)
        return img, 1
