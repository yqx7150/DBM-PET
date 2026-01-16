"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import torch as th
import numpy as np
import torchvision.utils as vutils
import torch.distributed as dist
import torch.nn.functional as F
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid
from ddbm import dist_util, logger
from ddbm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torch.fft
import torchmetrics
import tensorflow as tf
import scipy
from scipy import io

# from torchmetrics.image import StructuralSimilarityIndexMeasure, InceptionScore, PeakSignalNoiseRatio
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from torchmetrics import StructuralSimilarityIndexMeasure, MeanSquaredError
from ddbm.random_util import get_generator
from ddbm.karras_diffusion import karras_sample, forward_sample
from datasets import load_data
from scipy.io import savemat
from pathlib import Path
from PIL import Image
import lpips
# from skimage import img_as_float
# import brisque
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import cv2
import astra
import pydicom
import os
# from pytorch_fid import fid_score
# import niqe
def get_workdir(exp):
    workdir = f'./workdir/{exp}'
    return workdir
def compute_nps(image):
    """
    计算图像的噪声功率谱 (NPS)
    输入：image - 图像数据 (PyTorch Tensor)
    输出：nps_score - 图像的噪声功率谱值
    """
    # 转换为频域 (使用 PyTorch FFT)
    f = th.fft.fftn(image)
    fshift = th.fft.fftshift(f)
    magnitude_spectrum = th.abs(fshift)

    # 计算噪声功率谱
    nps_score = th.mean(magnitude_spectrum ** 2)  # 均方幅度谱

    return nps_score.item()  # 返回标量值



def nr_ssim(image, window_size=11, dynamic_range=255, K=(0.01, 0.03)):
    """
    Compute a no-reference SSIM (NR-SSIM) index for an image.

    Args:
        image (ndarray): The input image to evaluate.
        window_size (int): The size of the Gaussian filter window.
        dynamic_range (float): The dynamic range of the image (typically 255 for 8-bit images).
        K (tuple): Constants (C1, C2) for stabilizing SSIM calculation.

    Returns:
        nr_ssim_index (float): The NR-SSIM index.
    """
    C1 = (K[0] * dynamic_range) ** 2
    C2 = (K[1] * dynamic_range) ** 2

    image = image.astype(np.float64)

    # Compute local means
    mu = gaussian_filter(image, window_size / 6.0)

    # Compute local variances
    mu_sq = mu ** 2
    sigma_sq = gaussian_filter(image ** 2, window_size / 6.0) - mu_sq
    sigma = np.sqrt(sigma_sq)

    # NR-SSIM formula (self-similarity, hence sigma12 = sigma_sq)
    sigma12 = sigma_sq

    # Numerator and denominator for NR-SSIM map calculation
    numerator1 = (2 * mu * mu + C1)
    numerator2 = (2 * sigma12 + C2)
    denominator1 = (mu_sq + C1)
    denominator2 = (sigma_sq + C2)

    # Compute NR-SSIM map
    nr_ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)

    # Return the average NR-SSIM index
    nr_ssim_index = np.mean(nr_ssim_map)
    

    return nr_ssim_index

vol_geom = astra.create_vol_geom((128,128))
proj_geom = astra.create_proj_geom('parallel', 0.4, 512, np.linspace(0, 1 * np.pi,512,False))
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)


def min_max_normalization(x):
    """
    对输入的PyTorch张量进行最小-最大归一化操作，将其值映射到[0, 1]区间。

    """
    # 获取张量的最小值和最大值
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min)


    return x

def sino_to_pet(sinogram):
    "使用astra的FBP算法将PET成像结果转为正弦图"

    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = proj_id
    cfg['option'] = {}
    cfg['option']['FilterType'] = 'hamming'

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    x = astra.data2d.get(rec_id)

    return x

# 将归一化后的图像数据转换为合适的类型和模式（这里以单通道和三通道为例进行处理）
def convert_image_for_saving(img):
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):  # 单通道情况（灰度图像）
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img).convert('L')
    elif len(img.shape) == 3 and img.shape[2] == 3:  # 三通道情况（彩色图像）
        img = np.clip(img, 0, 1)  # 确保像素值在 0 - 1 范围
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img).convert('RGB')
    else:
        raise ValueError("Unsupported image format for conversion")
    return img


def main():
    args = create_argparser().parse_args()
    workdir = os.path.dirname(args.model_path)
    print(f"Model path: {args.model_path}")

    # Assume ema ckpt format: ema_{rate}_{steps}.pt
    split = args.model_path.split("_")
    print(f"Split path: {split}")

    step = int(split[-1].split(".")[0])
    sample_dir = Path(workdir) / f'sample_{step}/w={args.guidance}_churn={args.churn_step_ratio}'
    # Create sample directory
    sample_dir.mkdir(parents=True, exist_ok=True)

    logger.configure(dir=workdir)
    logger.log("creating model and diffusion...")

    # Create model and diffusion
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
    )

    # Load model state
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    model = model.to(th.device('cuda' if th.cuda.is_available() else 'cpu'))

    if args.use_fp16:
        model.convert_to_fp16()

    model.eval()
    logger.log("sampling...")

    all_images = []
    all_x0_images = []  # To store x0 images
    all_y0_images = []  # To store y0 images

    # Load data
    all_dataloaders = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        include_test=True,
        seed=args.seed,
        # num_workers=args.num_workers,
    )

    # Select appropriate dataloader based on split
    if args.split == 'train':
        dataloader = all_dataloaders[1]
    elif args.split == 'test':
        dataloader = all_dataloaders[2]
    else:
        raise NotImplementedError

    args.num_samples = len(dataloader.dataset)

    # 创建保存图像的子目录
    sample_images_dir = os.path.join(sample_dir, 'sample_images')
    x0_images_dir = os.path.join(sample_dir, 'x0_images')
    y0_images_dir = os.path.join(sample_dir, 'y0_images')
    
    sample_image_dir = os.path.join(sample_dir, 'sample_image')
    x0_img_dir = os.path.join(sample_dir, 'x0_img')
    y0_img_dir = os.path.join(sample_dir, 'y0_img')

    # 确保目录存在
    os.makedirs(sample_images_dir, exist_ok=True)
    os.makedirs(x0_images_dir, exist_ok=True)
    os.makedirs(y0_images_dir, exist_ok=True)

    os.makedirs(sample_image_dir, exist_ok=True)
    os.makedirs(x0_img_dir, exist_ok=True)
    os.makedirs(y0_img_dir, exist_ok=True)

    mse_metric = MeanSquaredError()
    ssim_metric = StructuralSimilarityIndexMeasure()


    # Assuming these are already initialized
    psnr_list, mse_list, ssim_list = [], [], []
    nps_list, lpips_list, nr_ssim_list = [], [], []
    all_images, all_x0_images, all_y0_images = [], [], []

    # For LPIPS
    lpips_model = lpips.LPIPS(net='alex').cuda()  # Load LPIPS model


    output_file = os.path.join(sample_dir, 'metrics_results.txt')
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            f.write("Batch\t    PSNR\t  MSE\t   SSIM\t  LPISPS\t    NPS\t   NR_SSIM\n")  # Write header

    for i, data in enumerate(dataloader):
        x0_image = data[0]
        x0 = x0_image.to(th.device('cuda' if th.cuda.is_available() else 'cpu'))
        y0_image = data[1]
        y0 = y0_image.to(th.device('cuda' if th.cuda.is_available() else 'cpu'))
        model_kwargs = {'xT': y0}
        index = data[2].to(th.device('cuda' if th.cuda.is_available() else 'cpu'))

        sample, path, nfe = karras_sample(
            diffusion,
            model,
            y0,
            x0,
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=th.device('cuda' if th.cuda.is_available() else 'cpu'),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=diffusion.sigma_min,
            sigma_max=diffusion.sigma_max,
            churn_step_ratio=args.churn_step_ratio,
            rho=args.rho,
            guidance=args.guidance
        )


        sample = sample.detach().cpu()  # Ensure the tensor is on CPU for numpy operations

        sample_min = sample.min()
        sample_max = sample.max()

        sample = (sample - sample_min) / (sample_max - sample_min)


        gathered_samples = sample.clone()

        mask = x0 == 0  # 创建一个布尔掩码，标记出 x0 中值为0的像素
        sample[mask] = 0  # 将 sample 中对应位置的像素值设为0，以消除那些在目标图像中为0的区域

        x0_np = x0_image.detach().cpu().numpy().squeeze()  # Remove batch and channel dimensions
        sample_np = sample.detach().cpu().numpy().squeeze()  # Remove batch and channel dimensions

        current_psnr = psnr(x0_np, sample_np, data_range=1)

        # Compute MSE
        current_mse = mse_metric(x0_image.clone().detach(), sample.clone().detach())

        # Compute SSIM
        current_ssim = ssim(x0_np, sample_np, data_range=sample_np.max() - sample_np.min())

        # LPIPS (Perceptual similarity)
        sample_lpips = sample.clone().detach().cuda()
        x0_lpips = x0_image.clone().detach().cuda()
        lpips_score = lpips_model(sample_lpips, x0_lpips).mean().item()


        nps_score = compute_nps(sample)

        nr_ssim_score = nr_ssim(sample.detach().cpu().numpy())

        # Append results to lists
        psnr_list.append(current_psnr)
        mse_list.append(current_mse)
        ssim_list.append(current_ssim)
        lpips_list.append(lpips_score)
        # niqe_list.append(niqe_score)
        nps_list.append(nps_score)
        nr_ssim_list.append(nr_ssim_score)

        # Log batch results
        with open(output_file, 'a') as f:
            f.write(
                f"{i}\t{current_psnr:.4f}\t{current_mse:.4f}\t{current_ssim:.4f}\t{lpips_score:.4f}\t{nps_score:.4f}\t{nr_ssim_score:.4f}\n")
        # Log the metrics
        print(
            f"Batch {i} PSNR: {current_psnr:.4f} MSE: {current_mse:.4f} SSIM: {current_ssim:.4f} LPIPS: {lpips_score:.4f} NPS: {nps_score:.4f} NR-SSIM: {nr_ssim_score:.4f}")
        vutils.save_image(sample,
                          f'{sample_images_dir}/fdg_k2_{i}.png',
                          )
        vutils.save_image(x0_image,
                          f'{x0_images_dir}/x0_batch_{i}.png')

        vutils.save_image(y0_image,
                          f'{y0_images_dir}/y0_batch_{i}.png')

        all_images.append(gathered_samples.detach().cpu().numpy())
        all_x0_images.append(x0_image.detach().cpu().numpy())
        all_y0_images.append(y0_image.detach().cpu().numpy())


        # 创建保存图像的子目录
        sample_images_mat_dir = os.path.join(sample_dir, 'sample_images_mat')
        x0_images_mat_dir = os.path.join(sample_dir, 'x0_images_mat')
        y0_images_mat_dir = os.path.join(sample_dir, 'y0_images_mat')

        os.makedirs(sample_images_mat_dir, exist_ok=True)
        os.makedirs(x0_images_mat_dir, exist_ok=True)
        os.makedirs(y0_images_mat_dir, exist_ok=True)

        # 分别保存每个图像到不同的 .mat 文件
        sample_mat_path = os.path.join(sample_images_mat_dir, f'fdg_k2_{1621+i}.mat')
        x0_mat_path = os.path.join(x0_images_mat_dir, f'x0_batch_{i}.mat')
        y0_mat_path = os.path.join(y0_images_mat_dir, f'y0_batch_{i}.mat')

        # 处理样本图像，去除 batch_size 和 channels，仅保留 height 和 width
        sample_np = sample.detach().cpu().numpy()
        x0_np = x0_image.detach().cpu().numpy()
        y0_np = y0_image.detach().cpu().numpy()  # 将 y0 图像转换到 [0, 1] 范围

        # 选择切片，仅保留 height 和 width
        # 假设样本是形状 [batch_size, channels, height, width]
        height, width = sample_np.shape[2], sample_np.shape[3]

        # 这里以第一个样本为例，如果要保存每个样本，可以使用循环
        sample_np = sample_np[0].transpose(1, 2, 0)  # 转换为 [height, width, channels]
        x0_np = x0_np[0].transpose(1, 2, 0)  # 转换为 [height, width, channels]
        y0_np = y0_np[0].transpose(1, 2, 0)  # 转换为 [height, width, channels]

        # 这里将通道维度去掉，确保保存为 [height, width]
        sample_np = np.squeeze(sample_np)  # 去掉单通道
        x0_np = np.squeeze(x0_np)  # 去掉单通道
        y0_np = np.squeeze(y0_np)  # 去掉单通道

        # 保存到 .mat 文件
        savemat(sample_mat_path, {
            'data': sample_np
        })

        savemat(x0_mat_path, {
            'data': x0_np
        })

        savemat(y0_mat_path, {
            'data': y0_np
        })
        sample_img = sino_to_pet(sample_np)
        x0_img = sino_to_pet(x0_np)
        y0_img = sino_to_pet(y0_np)
        sample_img = min_max_normalization(sample_img)
        x0_img = min_max_normalization(x0_img)
        y0_img = min_max_normalization(y0_img)
        # Compute PSNR
        current_img_psnr = psnr(x0_img, sample_img, data_range=1)

        # Compute MSE
        current_img_mse = np.mean((x0_img - sample_img) ** 2)


        # Compute SSIM
        current_img_ssim = ssim(x0_img, sample_img, data_range=sample_img.max() - sample_img.min())


        psnr_img_list, mse_img_list, ssim_img_list = [], [], []

        # Append results to lists
        psnr_img_list.append(current_img_psnr)
        mse_img_list.append(current_img_mse)
        ssim_img_list.append(current_img_ssim)


        output_file_img = os.path.join(sample_dir, 'metrics_img_results.txt')
        if not os.path.exists(output_file_img):
            with open(output_file_img, 'w') as f:
                f.write("Batch\t    PSNR\t  MSE\t   SSIM\n")  # Write header

        # Log batch results
        with open(output_file_img, 'a') as f:
            f.write(
                f"{i}\t{current_img_psnr:.4f}\t{current_img_mse:.4f}\t{current_img_ssim:.4f}\n")
        # Log the metrics
        print(
            f"Batch {i}_img PSNR: {current_img_psnr:.4f} MSE: {current_img_mse:.4f} SSIM: {current_img_ssim:.4f}")
        # 转换 sample_img
        sample_img = convert_image_for_saving(sample_img)
        # 转换 x0_img
        x0_img = convert_image_for_saving(x0_img)
        # 转换 y0_img
        y0_img = convert_image_for_saving(y0_img)

        # 保存 sample_img
        sample_image_path  = os.path.join(sample_image_dir, f"sample_img_{i}.png")
        sample_img.save(sample_image_path )

        # 保存 x0_img
        x0_img_path = os.path.join(x0_img_dir, f"x0_img_{i}.png")
        x0_img.save(x0_img_path)

        # 保存 y0_img
        y0_img_path = os.path.join(y0_img_dir, f"y0_img_{i}.png")
        y0_img.save(y0_img_path)

    # Calculate average metrics after all batches
    average_psnr = np.mean(psnr_list)
    average_mse = np.mean(mse_list)
    average_ssim = np.mean(ssim_list)
    average_lpips = np.mean(lpips_list)
    # average_niqe = np.mean(niqe_list)
    average_nps = np.mean(nps_list)
    average_nr_ssim = np.mean(nr_ssim_list)

    # Log average results at the end of the file

    with open(output_file, 'a') as f:
        f.write(
            f"\nAverage:\t{average_psnr:.4f}\t{average_mse:.4f}\t{average_ssim:.4f}\t{average_lpips:.4f}\t{average_nps:.4f}\t{average_nr_ssim:.4f}\n")

    print("Metrics saved to", output_file)
    # 合并并保存所有图像
    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(sample_dir, f"samples_{shape_str}_nfe{nfe}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr)

    logger.log("sampling complete")






def create_argparser():
    defaults = dict(
        data_dir="",  # only used in bridge
        dataset='edges2handbags',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        split='train',
        churn_step_ratio=0.,
        rho=7.0,
        steps=40,
        model_path="",
        exp="",
        seed=42,
        ts="",
        upscale=False,
        num_workers=2,
        guidance=1.,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()





