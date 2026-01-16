import os.path
import torch
import random
import numpy as np
import torchvision.transforms as transforms
# from .image_folder import make_dataset
from PIL import Image
import scipy
from scipy import io
import torchvision
import blobfile as bf
import re
from glob import glob
import astra
import pydicom
import os
# import multiprocessing
# multiprocessing.set_start_method('spawn')
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

def make_dataset(dir):
    """Make a dataset from the directory."""
    images = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith('.mat'):  # Ensure it collects only .mat files
                images.append(os.path.join(root, file))
    return images

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __scale(img, target_width, method=Image.BICUBIC):
    if isinstance(img, torch.Tensor):
        return torch.nn.functional.interpolate(img.unsqueeze(0), size=(target_width, target_width), mode='bicubic', align_corners=False).squeeze(0)
    else:
        return img.resize((target_width, target_width), method)

def __flip(img, flip):
    if flip:
        if isinstance(img, torch.Tensor):
            return img.flip(-1)
        else:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_flip(img, flip):
    return __flip(img, flip)

# 创建
vol_geom = astra.create_vol_geom((128,128))
proj_geom = astra.create_proj_geom('parallel', 0.4, 540, np.linspace(0, 1 * np.pi,540,False))
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)


def padding_img(img):
    w,h = img.shape
    h1 = 576
    tmp = np.zeros([h1,h1])
    x_start = int((h1 -w)//2)
    y_start = int((h1 -h)//2)
    tmp[x_start:x_start+w,y_start:y_start+h] = img
    return tmp

def pet_to_sino(image):
    "使用astra将PET成像结果转为正弦图"

    sinogram_id, sinogram = astra.create_sino(image, proj_id)
    astra.data2d.delete(sinogram_id)

    return sinogram
def normalize_frame(frame):
    # 获取帧的最小值和最大值
    min_val = np.min(frame)
    max_val = np.max(frame)

    # 防止除以零的情况
    if max_val > min_val:
        normalized_frame = (frame - min_val) / (max_val - min_val)
    else:
        normalized_frame = np.zeros_like(frame)  # 所有值相等时设置为零

    return normalized_frame

class EdgesDataset(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, train=True,  img_size=576, random_crop=False, random_flip=True):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        if train:
            self.train_dir = os.path.join(dataroot, 'train')  # get the image directory
            # self.train_paths = make_dataset(self.train_dir)  # get image paths
            self.A_paths = make_dataset(os.path.join(self.train_dir, 'fdg_3D'))
            self.K1_paths = make_dataset(os.path.join(self.train_dir, 'k1'))  # 获取 K1 路径
            self.K2_paths = make_dataset(os.path.join(self.train_dir, 'k2'))  # 获取 K2 路径
            self.K3_paths = make_dataset(os.path.join(self.train_dir, 'k3'))  # 获取 K3 路径
            self.K4_paths = make_dataset(os.path.join(self.train_dir, 'k4'))  # 获取 K4 路径

            self.A_paths = sorted(self.A_paths)
            self.K1_paths = sorted(self.K1_paths)
            self.K2_paths = sorted(self.K2_paths)
            self.K3_paths = sorted(self.K3_paths)
            self.K4_paths = sorted(self.K4_paths)

            print("A_paths:", os.path.join(self.train_dir, 'fdg_3D'))
            print("K1_paths:", os.path.join(self.train_dir, 'k1'))
            print("K2_paths:", os.path.join(self.train_dir, 'k2'))
            print("K3_paths:", os.path.join(self.train_dir, 'k3'))
            print("K4_paths:", os.path.join(self.train_dir, 'k4'))
            print(f'Number of training A_paths: {len(self.A_paths)}')
            print(f'Number of training K1_paths: {len(self.K1_paths)}')
            print(f'Number of training K2_paths: {len(self.K2_paths)}')
            print(f'Number of training K3_paths: {len(self.K3_paths)}')
            print(f'Number of training K4_paths: {len(self.K4_paths)}')
        else:

            self.test_dir = os.path.join(dataroot, 'val')  # get the image directory

            self.A_paths = make_dataset(os.path.join(self.test_dir, 'fdg_3D'))  # get image paths
            self.K1_paths = make_dataset(os.path.join(self.test_dir, 'k1'))  # get image paths
            self.K2_paths = make_dataset(os.path.join(self.test_dir, 'k2'))  # get image paths
            self.K3_paths = make_dataset(os.path.join(self.test_dir, 'k3'))  # get image paths
            self.K4_paths = make_dataset(os.path.join(self.test_dir, 'k4'))  # get image paths

            self.A_paths = self.sort_paths_by_suffix(self.A_paths)
            self.K1_paths = self.sort_paths_by_suffix(self.K1_paths)
            self.K2_paths = self.sort_paths_by_suffix(self.K2_paths)
            self.K3_paths = self.sort_paths_by_suffix(self.K3_paths)
            self.K4_paths = self.sort_paths_by_suffix(self.K4_paths)
            print("A_paths:", os.path.join(self.test_dir, 'fdg_3D'))
            print("K1_paths:", os.path.join(self.test_dir, 'k1'))
            print("K2_paths:", os.path.join(self.test_dir, 'k2'))
            print("K3_paths:", os.path.join(self.test_dir, 'k3'))
            print("K4_paths:", os.path.join(self.test_dir, 'k4'))
            print(f'Number of test A_paths: {len(self.A_paths)}')
            print(f'Number of test K1_paths: {len(self.K1_paths)}')
            print(f'Number of test K2_paths: {len(self.K2_paths)}')
            print(f'Number of test K3_paths: {len(self.K3_paths)}')
            print(f'Number of test K4_paths: {len(self.K4_paths)}')

        self.crop_size = img_size
        self.resize_size = img_size

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.train = train

        # test

    def sort_paths_by_suffix(self, paths):
        def get_suffix(path):
            # 提取文件名
            filename = os.path.basename(path)
            # 提取后缀数字
            match = re.search(r'_(\d+)\.mat$', filename)
            if match:
                return int(match.group(1))
            return float('inf')  # 如果没有匹配到后缀数字，设为无穷大

        # 根据后缀数字排序
        return sorted(paths, key=get_suffix)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # A是input，需要提取第一帧图像;B是label，对应K1,K2,K3,K4
        A_path = self.A_paths[index]
        K1_path = self.K1_paths[index]
        # K2_path = self.K2_paths[index]
        # K3_path = self.K3_paths[index]


        A_data = scipy.io.loadmat(A_path)
        K1_data = scipy.io.loadmat(K1_path)
        # K2_data = scipy.io.loadmat(K2_path)
        # K3_data = scipy.io.loadmat(K3_path)


        A_frames = A_data['data']
        K1_frame = K1_data['data_new']
        # K2_frame = K2_data['data_new']
        # K3_frame = K3_data['data_new']


        # 提取第一帧
        frame1_A = A_frames[:, :, 0]  # 第一帧
        frame2_A = A_frames[:, :, 1]  # 第2帧
        frame3_A = A_frames[:, :, 2]  # 第3帧
      

        sino1 = pet_to_sino(frame1_A)
        sino2 = pet_to_sino(frame2_A)
        sino3 = pet_to_sino(frame3_A)
        

        sinogram1 = padding_img(sino1).astype(np.float32)
        sinogram2 = padding_img(sino2).astype(np.float32)
        sinogram3 = padding_img(sino3).astype(np.float32)
       

        # 对每一帧进行归一化
        frame1_A_normalized = normalize_frame(sinogram1)
        frame2_A_normalized = normalize_frame(sinogram2)
        frame3_A_normalized = normalize_frame(sinogram3)
     
        # 转换为图像对象
        # A_image = Image.fromarray(first_frame_A)
        A1_image = Image.fromarray(frame1_A_normalized)
        A2_image = Image.fromarray(frame2_A_normalized)
        A3_image = Image.fromarray(frame3_A_normalized)

        sino4 = pet_to_sino(K1_frame)
        # sino4 = pet_to_sino(K2_frame)
        # sino4 = pet_to_sino(K3_frame)
        sinogram4 = padding_img(sino4).astype(np.float32)

        K1_frame_normalized = normalize_frame(sinogram4)
        # K2_frame_normalized = normalize_frame(sinogram4)
        # K3_frame_normalized = normalize_frame(sinogram4)

        K1_image = Image.fromarray(K1_frame_normalized)
        # K2_image = Image.fromarray(K2_frame_normalized)
        # K3_image = Image.fromarray(K3_frame_normalized)

        params = get_params(A1_image.size, self.resize_size, self.crop_size)

        transform_image = get_transform(params, self.resize_size, self.crop_size, crop=self.random_crop,
                                        flip=self.random_flip)

        A1_image = transform_image(A1_image)
        A2_image = transform_image(A2_image)
        A3_image = transform_image(A3_image)
      

        K1_image = transform_image(K1_image)
       
        if not self.train:
            # return A1_image, A2_image, A3_image, index, A_path
            return K1_image, K1_image, K1_image, A1_image, A2_image, A3_image, index, A_path
        else:
            # return A1_image, A2_image, A3_image, index
            return K1_image, K1_image, K1_image, A1_image, A2_image, A3_image, index

    def __len__(self):
        """返回数据集中图像的总数。"""
        return len(self.A_paths)  # 返回 A 图像路径的长度

class DIODE(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, train=True,  img_size=256, random_crop=False, random_flip=True, down_sample_img_size = 0, cache_name='cache', disable_cache=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        self.image_root = os.path.join(dataroot, 'train' if train else 'val')
        self.crop_size = img_size
        self.resize_size = img_size
        
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.train = train

        self.filenames = [l for l in os.listdir(self.image_root) if not l.endswith('.pth') and not l.endswith('_depth.png') and not l.endswith('_normal.png')]

        self.cache_path = os.path.join(self.image_root, cache_name+f'_{img_size}.pth')
        if os.path.exists(self.cache_path) and not disable_cache:
            self.cache = torch.load(self.cache_path)
            # self.cache['img'] = self.cache['img'][:256]
            self.scale_factor = self.cache['scale_factor']
            print('Loaded cache from {}'.format(self.cache_path))
        else:
            self.cache = None

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        
        fn = self.filenames[index]
        img_path = os.path.join(self.image_root, fn)
        label_path = os.path.join(self.image_root, fn[:-4]+'_normal.png')

        with bf.BlobFile(img_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        with bf.BlobFile(label_path, "rb") as f:
            pil_label = Image.open(f)
            pil_label.load()
        pil_label = pil_label.convert("RGB")

        # apply the same transform to both A and B
        params =  get_params(pil_image.size, self.resize_size, self.crop_size)

        transform_label = get_transform(params, self.resize_size, self.crop_size, method=Image.NEAREST, crop =False, flip=self.random_flip)
        transform_image = get_transform( params, self.resize_size, self.crop_size, crop =False, flip=self.random_flip)

        cond = transform_label(pil_label)
        img = transform_image(pil_image)

        # if self.down_sample_img:
        #     image_pil = np.array(image_pil).astype(np.uint8)
        #     down_sampled_image = self.down_sample_img(image=image_pil)["image"]
        #     down_sampled_image = get_tensor()(down_sampled_image)
        #     # down_sampled_image = transforms.ColorJitter(brightness = [0.85,1.15], contrast=[0.95,1.05], saturation=[0.95,1.05])(down_sampled_image)
        #     data_dict = {"ref":label_tensor, "low_res":down_sampled_image, "ref_ori":label_tensor_ori, "path": path}

        #     return image_tensor, data_dict
        if not self.train:
            return img, cond, index, fn
        else:
            return img, cond, index
        
    

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.cache is not None:
            return len(self.cache['img'])
        else:
            return len(self.filenames)
    

