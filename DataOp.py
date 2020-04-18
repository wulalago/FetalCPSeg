import os
import glob

import nibabel as nib
import numpy as np

import torch
from torch.utils.data import Dataset

from volumentations import *


def get_list(dir_path):
    """
    This function is to read data from data dir.
    The data dir should be set as follow:
    -- Data
        -- case1
            -- image.nii.gz
            -- label.nii.gz
        -- case2
        ...
    """
    print("Reading Data...")
    dict_list = []

    path_list = glob.glob(os.path.join(dir_path, '*'))

    path_list.sort()

    image_name = 'image.nii.gz'
    label_name = 'label.nii.gz'

    for path in path_list:
        dict_list.append(
            {
                'image_path': os.path.join(path, image_name),
                'label_path': os.path.join(path, label_name),
            }
        )

    # we split the data set to train set(0.75), val set(0.05), test set(0.2)
    train_ratio = 0.75
    val_ratio = 0.8
    train_num = round(len(dict_list)*train_ratio)
    val_num = round(len(dict_list)*val_ratio)

    train_list = dict_list[:train_num] + dict_list[:train_num]
    val_list = dict_list[train_num:val_num]
    test_list = dict_list[val_num:]
    print("Finished! Train:{} Val:{} Test:{}".format(len(train_list), len(val_list), len(test_list)))

    return train_list, val_list, test_list


def get_augmentation():
    """
    here is the data augmentation compose function by packages volumentations:
    https://github.com/ashawkey/volumentations
    It provide a various augmentation strategy in 3D data
    """
    return Compose([
        # Flip(0),
        # Flip(1),
        # Flip(2),
        # RandomRotate90((0, 1)),
        # RandomRotate90((0, 2)),
        # RandomRotate90((1, 2))
    ], p=0.5)


class TrainGenerator(object):
    """
    This is the class to generate the patches
    """
    def __init__(self, data_list, batch_size, patch_size):
        self.data_list = data_list
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.aug = get_augmentation()

    def get_item(self):

        dict_list = random.sample(self.data_list, self.batch_size)

        image_list = [dict_item['image_path'] for dict_item in dict_list]
        label_list = [dict_item['label_path'] for dict_item in dict_list]

        image_patch, label_patch = self._sample_patch(image_list, label_list)

        return image_patch, label_patch

    def _sample_patch(self, image_list, clean_list):
        half_size = self.patch_size // 2
        image_patch_list = []
        label_patch_list = []

        for image_path, clean_path in zip(image_list, clean_list):
            image = nib.load(image_path).get_fdata()
            label = nib.load(clean_path).get_fdata()

            # here we augment the corresponding data and label
            data = {'image': image, 'label': label}
            aug_data = self.aug(**data)
            image, label = aug_data['image'], aug_data['label']

            w, h, d = image.shape

            label_index = np.where(label == 1)
            length_label = label_index[0].shape[0]

            p = random.random()
            # we set a probability(p) to make most of the center of sampling patches
            # locate to the regions with label not background
            if p < 0.875:
                sample_id = random.randint(1, length_label-1)
                x, y, z = label_index[0][sample_id], label_index[1][sample_id], label_index[2][sample_id]
            else:
                x, y, z = random.randint(0, w), random.randint(0, h), random.randint(0, d)

            # here we prevent the sampling patch overflow volume
            if x < half_size:
                x = half_size
            elif x > w-half_size:
                x = w-half_size-1

            if y < half_size:
                y = half_size
            elif y > h-half_size:
                y = h-half_size-1

            if z < half_size:
                z = half_size
            elif z > d-half_size:
                z = d-half_size-1

            image_patch = image[x-half_size:x+half_size, y-half_size:y+half_size, z-half_size:z+half_size].astype(np.float32)
            label_patch = label[x-half_size:x+half_size, y-half_size:y+half_size, z-half_size:z+half_size].astype(np.float32)

            image_patch_list.append(image_patch[np.newaxis, np.newaxis, ...])
            label_patch_list.append(label_patch[np.newaxis, np.newaxis, ...])

        image_out = np.concatenate(image_patch_list, axis=0)
        label_out = np.concatenate(label_patch_list, axis=0)

        return image_out, label_out

