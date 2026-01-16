import os.path
import torch
import random
import numpy as np
import torchvision.transforms as transforms
# from .image_folder import make_dataset
from PIL import Image
import scipy
from scipy.io import savemat

import torchvision
import blobfile as bf

from glob import glob
def make_dataset(dir):
    """Make a dataset from the directory."""
    images = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith('.mat'):  # Ensure it collects only .mat files
                images.append(os.path.join(root, file))
    return images

def get_params( size,  resize_size,  crop_size):
    w, h = size
    new_h = h
    new_w = w

    ss, ls = min(w, h), max(w, h)  # shortside and longside
    width_is_shorter = w == ss
    ls = int(resize_size * ls / ss)
    ss = resize_size
    new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(params,  resize_size,  crop_size, method=Image.BICUBIC,  flip=True, crop = True, totensor=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __scale(img, crop_size, method)))

    if flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if totensor:
        transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

dataroot=r'/home/world/CL/DDBM/edges2handbags'
train_dir = os.path.join(dataroot, 'train')  # get the image directory
# self.train_paths = make_dataset(self.train_dir)  # get image paths
A_paths = make_dataset(os.path.join(train_dir, 'fdg_3D'))
K1_paths = make_dataset(os.path.join(train_dir, 'k1'))  # 获取 K1 路径


A_paths = sorted(A_paths)
K1_paths = sorted(K1_paths)


print("A_paths:", os.path.join(train_dir, 'fdg_3D'))
print("K1_paths:", os.path.join(train_dir, 'k1'))

print(f'Number of training A_paths: {len(A_paths)}')
print(f'Number of training K1_paths: {len(K1_paths)}')

# crop_size = img_size
# resize_size = img_size

# random_crop = False
# random_flip = True

# 指定保存图像的目录
output_dir = r'/home/world/CL/DDBM/processed_images'
os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在则创建
for index in range(len(A_paths)):

    A_path = A_paths[index]
    K1_path = K1_paths[index]

    # 读取 MAT 文件
    A_data = scipy.io.loadmat(A_path)
    K1_data = scipy.io.loadmat(K1_path)


    # 假设 A_data 和 B_data 中存储的图像数据分别在 'A' 和 'B' 键下
    A_frames = A_data['data']
    K1_frame = K1_data['data_new']


    # 提取第一帧
    first_frame_A = A_frames[:, :, 0]  # 第一帧
    # # 数据归一化，模拟 MATLAB 的 mat2gray
    # min_val = np.min(first_frame_A)
    # max_val = np.max(first_frame_A)
    #
    # # 防止除以零的情况
    # if max_val > min_val:
    #     first_frame_A_normalized = (first_frame_A - min_val) / (max_val - min_val)
    # else:
    #     first_frame_A_normalized = np.zeros_like(first_frame_A)  # 所有值相等时设置为零

    # 将归一化后的数据乘以 255 转换为 uint8 格式
    # first_frame_A_normalized = (first_frame_A_normalized * 255).astype(np.float32)
    # first_frame_A_normalized = (first_frame_A_normalized * 255).astype(np.uint8)

    # # 处理 K1_frame 确保可以保存
    # min_val_K1 = np.min(K1_frame)
    # max_val_K1 = np.max(K1_frame)
    #
    # if max_val_K1 > min_val_K1:
    #     K1_frame_normalized = (K1_frame - min_val_K1) / (max_val_K1 - min_val_K1)
    # else:
    #     K1_frame_normalized = np.zeros_like(K1_frame)
    #
    # # 将 K1_frame 转换为 uint8 类型
    # K1_frame_normalized = (K1_frame_normalized * 255).astype(np.uint8)
    # K1_image = Image.fromarray(K1_frame_normalized)
    #
    # # 确保 K1_image 是可以保存的模式
    # if K1_image.mode != 'L' and K1_image.mode != 'RGB':
    #     K1_image = K1_image.convert('L')  # 转换为灰度模式
    # 转换为图像对象
    # A_image = Image.fromarray(first_frame_A_normalized)
    sample_mat_path = os.path.join(output_dir, f'sample_batch_{index}.mat')
    savemat(sample_mat_path, {
        'first_frame_A': first_frame_A  # 保存采样图像
    })
    # A_image = Image.fromarray(first_frame_A)
    # 确保是灰度图（如果需要的话）
    # if A_image.mode != 'L':
    #     A_image = A_image.convert('L')
    #
    # K1_image = Image.fromarray(K1_frame)

    # apply the same transform to both A and B


 # 保存图像
 #    A_image.save(os.path.join(output_dir, f'A_image_{index:03d}.png'))
    # K1_image.save(os.path.join(output_dir, f'K1_image_{index:03d}.jpg'))

    print(f'Saved A_image and K1_image for index {index}')