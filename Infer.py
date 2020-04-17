import os
import time

import nibabel as nib
import numpy as np

import torch

from Network import MixAttNet
from Utils import AvgMeter, check_dir
from DataOp import get_list


torch.cuda.set_device(0)

output_path = os.path.join('output/')

_, test_list = get_list(dir_path='Data/')
net = MixAttNet().cuda()
net.load_state_dict(torch.load(output_path+'/ckpt/best_val.pth.gz'))

patch_size = 64
spacing = 4

save_path = os.path.join(output_path, 'save_data')
check_dir(save_path)

net.eval()

test_meter = AvgMeter()

for idx, data_dict in enumerate(test_list):
    image_path = data_dict['image_path']

    image = nib.load(image_path).get_fdata()

    w, h, d = image.shape

    pre_count = np.zeros_like(image, dtype=np.float32)
    predict = np.zeros_like(image, dtype=np.float32)

    x_list = np.squeeze(np.concatenate((np.arange(0, w - patch_size, patch_size // spacing)[:, np.newaxis],
                                        np.array([w - patch_size])[:, np.newaxis])).astype(np.int))
    y_list = np.squeeze(np.concatenate((np.arange(0, h - patch_size, patch_size // spacing)[:, np.newaxis],
                                        np.array([h - patch_size])[:, np.newaxis])).astype(np.int))
    z_list = np.squeeze(np.concatenate((np.arange(0, d - patch_size, patch_size // spacing)[:, np.newaxis],
                                        np.array([d - patch_size])[:, np.newaxis])).astype(np.int))
    start_time = time.time()

    for x in x_list:
        for y in y_list:
            for z in z_list:
                image_patch = image[x:x + patch_size, y:y + patch_size, z:z + patch_size].astype(np.float32)
                patch_tensor = torch.from_numpy(image_patch[np.newaxis, np.newaxis, ...]).cuda()
                predict[x:x + patch_size, y:y + patch_size, z:z + patch_size] += net(patch_tensor).squeeze().cpu().data.numpy()
                pre_count[x:x + patch_size, y:y + patch_size, z:z + patch_size] += 1

    predict /= pre_count

    predict = np.squeeze(predict)
    image = np.squeeze(image)

    predict[predict > 0.5] = 1
    predict[predict < 0.5] = 0

    image_nii = nib.Nifti1Image(image, affine=None)
    predict_nii = nib.Nifti1Image(predict, affine=None)

    check_dir(os.path.join(save_path, '{}'.format(idx)))
    nib.save(image_nii, os.path.join(save_path, '{}/image.nii.gz'.format(idx)))
    nib.save(predict_nii, os.path.join(save_path, '{}/predict.nii.gz'.format(idx)))

    print("[{}] Testing Finished, Cost {:.2f}s".format(idx, time.time()-start_time))
