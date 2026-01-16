import argparse
import os
import numpy as np
import scipy.io as io
import torch
from scipy.io import loadmat
from scipy.integrate import cumtrapz, trapz
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import logging
import re
import tensorflow as tf
from natsort import natsorted
import astra
import gc

# from dataset.pet_dataset_multiple_output_noise_fdg import PetDataset
# from model.model_new import MultiOutputReversibleGenerator
from untils.com_untils_test import compare_psnr_show_save
from untils.com_untils_test import get_mean_k_data, save_img, calculate_img_np

# from untils.utils_metric import calculate_ms_ssim, calculate_nrmse

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
# 假设你希望从这些路径中读取图像文件
class PetDataset(Dataset):
    def __init__(self, root_folder):

        self.fdg_files = natsorted(os.listdir(os.path.join(root_folder, 'fdg_3D')))
        self.fdg_noise_files = natsorted(os.listdir(os.path.join(root_folder, 'fdg_3D_noise')))

        self.k1_files = natsorted(os.listdir(os.path.join(root_folder, 'pred_k1')))
        self.k2_files = natsorted(os.listdir(os.path.join(root_folder, 'pred_k2')))
        self.k3_files = natsorted(os.listdir(os.path.join(root_folder, 'pred_k3')))
        self.k4_files = natsorted(os.listdir(os.path.join(root_folder, 'pred_k4')))
        self.ki_files = natsorted(os.listdir(os.path.join(root_folder, 'ki')))
        self.vb_files = natsorted(os.listdir(os.path.join(root_folder, 'vb')))
        self.root_folder = root_folder
    
    def __len__(self):
        return len(self.fdg_files)

    def __getitem__(self, idx):
        fdg_data = loadmat(os.path.join(self.root_folder, 'fdg_3D', self.fdg_files[idx]))['data']
        fdg_noise_data = loadmat(os.path.join(self.root_folder, 'fdg_3D_noise', self.fdg_noise_files[idx]))['data']

        k1_data = loadmat(os.path.join(self.root_folder, 'pred_k1', self.k1_files[idx]))['sample_img']
        k2_data = loadmat(os.path.join(self.root_folder, 'pred_k2', self.k2_files[idx]))['sample_img']
        k3_data = loadmat(os.path.join(self.root_folder, 'pred_k3', self.k3_files[idx]))['sample_img']
        k4_data = loadmat(os.path.join(self.root_folder, 'pred_k4', self.k4_files[idx]))['data_new']
        ki_data = loadmat(os.path.join(self.root_folder, 'ki', self.ki_files[idx]))['data']
        vb_data = loadmat(os.path.join(self.root_folder, 'vb', self.vb_files[idx]))['data']

        k1_data_new = np.tile(k1_data[:, :, np.newaxis], 3)
        k2_data_new = np.tile(k2_data[:, :, np.newaxis], 3)
        k3_data_new = np.tile(k3_data[:, :, np.newaxis], 3)
        k4_data_new = np.tile(k4_data[:, :, np.newaxis], 3)
        # k4_data_new = k3_data_new

        return fdg_data,fdg_noise_data, k1_data_new, k2_data_new, k3_data_new, k4_data_new, ki_data, vb_data
# 创建
vol_geom = astra.create_vol_geom((128,128))
proj_geom = astra.create_proj_geom('parallel', 0.4, 540, np.linspace(0, 1 * np.pi,540,False))
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)

def pet_to_sino(image):
    "使用astra将PET成像结果转为正弦图"

    sinogram_id, sinogram = astra.create_sino(image, proj_id)
    astra.data2d.delete(sinogram_id)

    return sinogram

def sino_to_pet(sinogram):
    "使用astra的FBP算法将PET成像结果转为正弦图"
    # sinogram = sinogram.squeeze(0).squeeze(0).cpu().numpy()
    # sinogram = sinogram.squeeze().cpu().numpy()

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

    # 显式清理ASTRA资源
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(rec_id)
    astra.algorithm.delete(alg_id)
    return x

def main():
    # ======================================define the model============================================
    # 配置信息
    CP_PATH = '/home/un/world/DDBM3/edges2handbags/CP/CP_FDG.mat'
    sampling_intervals = [30, 30, 30, 30, 120, 120, 120, 120, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
    cp_data = loadmat(CP_PATH)['CP_FDG']
    cp_data = cp_data[:, :3600]
    root2 = '/home/un/world/DDBM/edges2handbags/val/'

    # 初始化 Dataset
    dataset = PetDataset(root_folder=root2)

    # 创建 DataLoader
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    SSIM_FDG = []
    MS_SSIM_FDG = []
    MSE_FDG = []
    NRMSE_FDG = []


    checkpoint = 99
    ckpt_allname = '{:04d}.pth'.format(checkpoint)

    save_path = '/home/un/world/DDBM/test/401fbp'

    log_dir = '/home/un/world/DDBM/test/401fbp/'
    log_file = os.path.join(log_dir, 'psnr_log.txt')

    # 检查并创建目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 配置日志记录
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    print("[INFO] Start test...")
    for i_batch, (
            fdg_batch,fdg_noise_batch, k1_data_batch, k2_data_batch, k3_data_batch, k4_data_batch, ki_data_btach,vb_data_batch) in enumerate(tqdm(test_dataloader)):  # tqdm是一个可以显示进度条的模块。enumerate()函数是python的内置函数，可以同时遍历lt中元素及其索引，i是索引，item是lt中的元素。
        # 组合k数据，torch.cat在dim=2上连接k1,k2,k3,k4得到形状为128*128*4的张量
        PSNR_FDG_IMG = []
        SSIM_FDG_IMG = []
        MSE_FDG_IMG = []
        NRMSE_FDG_IMG = []

        processed_frames = []
        for frame_idx in range(fdg_batch.shape[3]):  # 遍历18帧（假设fdg_batch形状为(1, 128, 128, 18)）
            # 提取当前帧数据（形状：128x128）
            frame = fdg_batch[0, :, :, frame_idx].cpu().numpy()

            # 投影转换为正弦图
            sino = pet_to_sino(frame)

            # 反投影回图像
            recon_frame = sino_to_pet(sino)
            # 立即清理中间变量
            del sino
            gc.collect()
            # 调整维度以匹配原数据格式（添加批次和通道维度）
            processed_frames.append(recon_frame[np.newaxis, :, :, np.newaxis])

        # 堆叠所有处理后的帧，恢复原始形状(1, 128, 128, 18)
        fdg_batch = np.concatenate(processed_frames, axis=3)
        fdg_batch = torch.from_numpy(fdg_batch).float()  # 转换为torch张量


        k_data_batch = torch.cat(
            (k1_data_batch.squeeze()[:, :, 0].unsqueeze(2), k2_data_batch.squeeze()[:, :, 0].unsqueeze(2),
             k3_data_batch.squeeze()[:, :, 0].unsqueeze(2),
             k4_data_batch.squeeze()[:, :, 0].unsqueeze(2)), dim=2)  # 128x128x4
        # print('k_data_batch',k_data_batch.shape)

        target_forward_fdg = fdg_batch.permute(0, 3, 1, 2).float().cuda()  #
        target_forward_fdg_label = fdg_batch.permute(0, 3, 1, 2).float().cuda()

        fdg_input = target_forward_fdg[:, 0:3, :, :]
        fdg_input = torch.cat([fdg_input, fdg_input], dim=1)

        # with torch.no_grad():  # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
        fdg_input = torch.abs(fdg_input)
        fdg_input = torch.clamp(fdg_input, 0, 1)  # torch.Size([1, 96, 128, 128])

        pred_fdg_k = k_data_batch.cpu().numpy()

        ## img
        # pred_data = calculate_img_torch(reconstruct_for, sampling_intervals, cp_data)
        pred_fdg_k = torch.from_numpy(pred_fdg_k)
        ## 将k1,k2,k3,k4转换为img：将模型预测的张量reconstruct_for和实际数据k_data_batch转换为图像数据
        pred_data = calculate_img_np(fdg_input, k_data_batch, sampling_intervals, cp_data)  # 128x128x18

        target_forward_fdg_label = target_forward_fdg_label.squeeze().permute(1, 2, 0)  # 128x128x18
        pred_data = pred_data[:, :, 3:18]
        target_forward_fdg_label = target_forward_fdg_label[:, :, 3:18].cpu().numpy()
        mask = target_forward_fdg_label == 0    #创建一个布尔掩码，标记出 target_forward_fdg_label 中值为0的像素
        pred_data[mask] = 0     #将 pred_data 中对应位置的像素值设为0，以消除那些在目标图像中为0的区域

        # psnr_fdg_frame = []  # 用来存储每一帧的 PSNR
        # 记录每一帧的 PSNR
        logging.info(f'pred_fdg_{i_batch + 1621}')  # 标记批次编号

        for i in range(15):
            
            psnr_fdg_img = compare_psnr(255 * abs(target_forward_fdg_label[:, :, i]), 255 * abs(pred_data[:, :, i]),data_range=255)  # abs()函数返回数字的绝对值。

            ssim_fdg_img = compare_ssim(abs(target_forward_fdg_label[:, :, i]), abs(pred_data[:, :, i]),data_range=1)

            mse_fdg_img = compare_mse(abs(target_forward_fdg_label[:, :, i]), abs(pred_data[:, :, i]))

            # ms_ssim_fdg_img = compare_ms_ssim(abs(target_forward_fdg_label[:, :, i]), abs(pred_data[:, :, i]))
            if not np.all(target_forward_fdg_label[:, :, i] == 0):
                nrmse_fdg_img = compare_nrmse(abs(target_forward_fdg_label[:, :, i]), abs(pred_data[:, :, i]))
            else:
                nrmse_fdg_img = 0
            logging.info(f"PSNR {i + 3} frame: {psnr_fdg_img}  SSIM {i + 3} frame: {ssim_fdg_img}  MSE {i + 3} frame: {mse_fdg_img}  NRMSE {i + 3} frame: {nrmse_fdg_img}")

            PSNR_FDG_IMG.append(psnr_fdg_img)
            SSIM_FDG_IMG.append(ssim_fdg_img)
            # MS_SSIM_FDG_IMG.append(ms_ssim_fdg_img)
            MSE_FDG_IMG.append(mse_fdg_img)
            NRMSE_FDG_IMG.append(nrmse_fdg_img)

            #  保存 result
            os.makedirs(save_path + '/img/pred_fdg_img',exist_ok=True) 
            os.makedirs(save_path + '/img/target_fdg_img', exist_ok=True)
            os.makedirs(save_path + '/img/pred_fdg_mat', exist_ok=True)
            os.makedirs(save_path + '/img/target_fdg_mat', exist_ok=True)

            save_img(target_forward_fdg_label[:, :, i],
                     save_path + '/img/target_fdg_img' + '/target_fdg_' + str(i_batch + 1621) + '_' + str(i + 3) + '.png')
            save_img(pred_data[:, :, i],
                     save_path + '/img/pred_fdg_img' + '/pred_fdg_' + str(i_batch + 1621) + '_' + str(i + 3) + '.png')
            # mat
            io.savemat(
                save_path + '/img/target_fdg_mat' + '/target_fdg_' + str(i_batch + 1621) + '_' + str(i + 3) + '.mat',
                {'data': target_forward_fdg_label[:, :, i]})
            io.savemat(
                save_path + '/img/pred_fdg_mat' + '/pred_fdg_' + str(i_batch + 1621) + '_' + str(i + 3) + '.mat',
                {'data': pred_data[:, :, i]})
        # 计算每个批次的平均值
        avg_psnr = np.mean(PSNR_FDG_IMG)
        avg_ssim = np.mean(SSIM_FDG_IMG)
        avg_mse = np.mean(MSE_FDG_IMG)
        avg_nrmse = np.mean(NRMSE_FDG_IMG)
        # 记录每个批次的平均值
        logging.info(f"Batch {i_batch + 1621} - Average PSNR: {avg_psnr}")
        logging.info(f"Batch {i_batch + 1621} - Average SSIM: {avg_ssim}")
        logging.info(f"Batch {i_batch + 1621} - Average MSE: {avg_mse}")
        logging.info(f"Batch {i_batch + 1621} - Average NRMSE: {avg_nrmse}")
        # 清理内存
        del fdg_batch, k_data_batch, target_forward_fdg, target_forward_fdg_label, fdg_input, pred_data
        gc.collect()
        torch.cuda.empty_cache()
    compare_psnr_show_save(PSNR_FDG_IMG, SSIM_FDG_IMG, MSE_FDG_IMG, NRMSE_FDG_IMG,"fdg", save_path, 2, ckpt_allname)


def compare_psnr_show_save(PSNR, SSIM, MSE, NRMSE, show_name, save_path, index, ckpt_allname):
    # print(show_name)
    # print(PSNR)
    ave_psnr = sum(PSNR) / len(PSNR)
    PSNR_std = np.std(PSNR)

    ave_ssim = sum(SSIM) / len(SSIM)
    SSIM_std = np.std(SSIM)

    # ave_ms_ssim = sum(MS_SSIM) / len(MS_SSIM)
    # MS_SSIM_std = np.std(MS_SSIM)

    ave_mse = sum(MSE) / len(MSE)
    MSE_std = np.std(MSE)

    ave_nrmse = sum(NRMSE) / len(NRMSE)
    NRMSE_std = np.std(NRMSE)
    if index == 1:
        file_name = 'k_results_test.txt'
        print('k_results:')
    elif index == 2:
        file_name = 'img_results_test.txt'
        print('img_results:')
    elif index == 3:
        file_name = 'ki_results_test.txt'
        print('ki_results:')
    else:
        file_name = 'vb_results_test.txt'
        print('vb_results:')
    print('ave_psnr_' + show_name, ave_psnr)
    print('ave_ssim_' + show_name, ave_ssim)
    # print('ave_ms_ssim_' + show_name, ave_ms_ssim)
    print('ave_mse_' + show_name, ave_mse)
    print('ave_nrmse_' + show_name, ave_nrmse)

    file_path = os.path.join(save_path, file_name)
    with open(file_path, 'a+') as f:
        f.write('\n' * 3)
        f.write(ckpt_allname + '_' + show_name + '\n')
                
        f.write('ave_psnr:' + str(ave_psnr) + ' ' * 3 + 'PSNR_std:' + str(PSNR_std) + '\n')

        f.write('ave_ssim:' + str(ave_ssim) + ' ' * 3 + 'SSIM_std:' + str(SSIM_std) + '\n')

        # f.write('ave_ms_ssim:' + str(ave_ms_ssim) + ' ' * 3 + 'SSIM_std:' + str(MS_SSIM_std) + '\n')

        f.write('ave_mse:' + str(ave_mse) + ' ' * 3 + 'MSE_std:' + str(MSE_std) + '\n')

        f.write('ave_nrmse:' + str(ave_nrmse) + ' ' * 3 + 'nrmse_std:' + str(NRMSE_std) + '\n')


def get_mean_k_data(reconstruct_for):
    """用于处理输入数组并返回四个平均图像数据
    # get_mean_k_data_np的tensor版本
    Args:
        reconstruct_for:

    Returns:

    """
    k1, k2, k3, k4 = torch.split(reconstruct_for, 3, dim=1)    #使用 torch.split 将输入张量沿第一个维度（通道维度）分割成四个部分，每部分包含12个特征通道。
    pred_k1 = torch.mean(k1.squeeze(), dim=0)       #k1.squeeze() 移除所有大小为1的维度，通常用来简化形状
    pred_k2 = torch.mean(k2.squeeze(), dim=0)       #torch.mean(..., dim=0) 沿第一个维度计算均值，得到一个形状为 (H, W) 的张量，表示平均图像
    pred_k3 = torch.mean(k3.squeeze(), dim=0)
    pred_k4 = torch.mean(k4.squeeze(), dim=0)
    return pred_k1, pred_k2, pred_k3, pred_k4


def calculate_img_np(reconstruct_for_k_data, k_data_batch, sampling_intervals, cp_data):
    """
            pred_data = calculate_img_np(fdg_input, k_data_batch, sampling_intervals, cp_data)  # 128x128x18

    用于生成基于模型预测的图像数据
    Args:
        reconstruct_for_k_data: 网络预测的结果
        sampling_intervals: 采样协议
        cp_data:血浆

    Returns:预测的18帧图像

    """
    # 由预测的k1-k4 图像和已知的Cp数据生成预测的18帧的数据
    reconstruct_for_k_data = reconstruct_for_k_data.cpu().detach().numpy()  # 128,128,12

    CP_FDG = cp_data
    pred_CT_FDG = update_tracer_concentration_np(reconstruct_for_k_data, k_data_batch, CP_FDG, 0)
    # 利用预测的reconstruct_for_k_data和k_data_batch数据以及血浆浓度CP_FDG，计算预测的tracer(示踪剂)浓度图像pred_CT_FDG
    f_FDG = np.zeros((128, 128, 18))

    lambda_value = np.log(2) / (109.8 * 60)     #计算一个衰减常数
    start_index = 0
    for k in range(len(sampling_intervals)):       #sampling_intervals: 这个列表包含了不同时间点的采样间隔
        end_index = start_index + sampling_intervals[k] - 1
        # print("start_index:", start_index)  # 确保 lambda_value 正确传递
        # print("end_index:", end_index)  # 确保 lambda_value 正确传递

        f_FDG[:, :, k] = calculate_xm(start_index, end_index, pred_CT_FDG, lambda_value)
        start_index = end_index + 1
    # print(f"f_FDG min: {np.min(f_FDG)}, f_FDG max: {np.max(f_FDG)}")
    # pred_fdg = normalize_array(f_FDG)
    pred_fdg = f_FDG / np.max(f_FDG)
    # print(f"pred_fdg min: {np.min(pred_fdg)}, pred_fdg max: {np.max(pred_fdg)}")

    # pred_fdg = f_FDG / np.max(f_FDG, axis=(0, 1), keepdims=True)  # 逐帧归一化
    # pred_fdg = torch.from_numpy(pred_fdg)
    return pred_fdg


# 用于处理输入数组 reconstruct_for 并返回四个平均图像数据
def get_mean_k_data_np(reconstruct_for):
    # print(reconstruct_for.shape)

    k1, k2, k3, k4 = np.split(reconstruct_for, 4, axis=1)
    # pred_k1 = np.mean(k1.squeeze().double(), axis=0)
    # pred_k2 = np.mean(k2.squeeze(), axis=0)
    # pred_k3 = np.mean(k3.squeeze(), axis=0)
    # pred_k4 = np.mean(k4.squeeze(), axis=0)
    pred_k1 = np.mean(np.squeeze(k1).astype(np.float64), axis=0).astype(np.float32) #np.squeeze(k1): 去除 k1 数组中所有单维度（(N, 1, H, W) 变为 (N, H, W)）
    pred_k2 = np.mean(np.squeeze(k2).astype(np.float64), axis=0).astype(np.float32) #.astype(np.float64): 将数组转换为 float64 类型，以确保计算精度
    pred_k3 = np.mean(np.squeeze(k3).astype(np.float64), axis=0).astype(np.float32) #np.mean(..., axis=0): 计算沿着第0轴（批量维度）的均值，得到每个像素的平均值，输出形状为 (H, W)
    pred_k4 = np.mean(np.squeeze(k4).astype(np.float64), axis=0).astype(np.float32) #.astype(np.float32): 将结果转换为 float32 类型，通常为了节省内存或符合特定要求
    return pred_k1, pred_k2, pred_k3, pred_k4



# def compare_ms_ssim(pred, target):
#     """
#     对数据求ms_ssim指标
#     Args:
#         pred:
#         target:
#
#     Returns:
#
#     """
#     pred = tf.convert_to_tensor(pred)
#     pred = tf.expand_dims(pred, axis=-1)
#
#     target = tf.convert_to_tensor(target)
#     target = tf.expand_dims(target, axis=-1)
#     pred = tf.cast(pred, dtype=tf.float64)
#     target = tf.cast(target, dtype=tf.float64)
#     max_val = 1
#
#     result = tf.image.ssim_multiscale(pred, target, max_val, filter_size=1)
#     result = result.numpy()
#     return result

def normalize_array(arr):
    """
    最大最小值归一化 对(128,128,18)
    Args:
        arr:

    Returns:

    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max == arr_min:
        # 分母为零，避免除以零的情况
        normalize_arr = np.zeros_like(arr)
    else:
        normalize_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalize_arr

# def normalize_array(arr):
#     """
#     对输入的数组进行最大值 - 最小值归一化操作
#     :param arr: 输入的 numpy 数组或 torch.Tensor
#     :return: 归一化后的 numpy 数组或 torch.Tensor
#     """
#     if isinstance(arr, torch.Tensor):
#         # 如果是 torch.Tensor 类型
#         if arr.max() == arr.min():
#             if arr.max() == 0:
#                 return arr
#             else:
#                 return torch.ones_like(arr)
#         return (arr - arr.min()) / (arr.max() - arr.min())
#     elif isinstance(arr, np.ndarray):
#         # 如果是 numpy.ndarray 类型
#         if np.max(arr) == np.min(arr):
#             if np.max(arr) == 0:
#                 return arr
#             else:
#                 return np.ones_like(arr)
#         return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
#     else:
#         raise TypeError("Input must be a numpy.ndarray or a torch.Tensor.")



def update_tracer_concentration_np(reconstruct_for_k_data, k_data_batch, cp_data, number):
    """
    由预测的k1,k2,k3,k4生成预测的TAC曲线
    Args:
        reconstruct_for_k_data:
        cp_data:
        number:

    Returns:

    """
    #从reconstruct_for_k_data中提取k1,k2,k3,k4
    # k1, k2, k3, k4 = get_mean_k_data_np(reconstruct_for_k_data)  # 128x128
    # print(k_data_batch.shape)
    # assert 0
    # print(k_data_batch.shape)
    target_k1, target_k2, target_k3, target_k4 = torch.split(k_data_batch, 1, dim=2)    #从k_data_batch中沿维度2分离数据
    # print(k2.shape)
    # print(k2_data_batch.shape)
    #
    # assert 0
    # k1, k2, k3, k4 = np.split(reconstruct_for_k_data, 4, axis=2)
    k1 = target_k1
    k2 = target_k2
    k3 = target_k3
    k4 = target_k4

    k1 = k1.squeeze()
    k2 = k2.squeeze()
    k3 = k3.squeeze()
    k4 = k4.squeeze()
    # print(k1.shape)
    # assert 0
    k4 = np.zeros((128, 128), dtype=np.float32) # 初始化和标准化 k 值

    #

    # # 进行最大值最小值归一化
    # if number == 1 :
    #     k1 = normalize_array(k1)
    #     k2 = normalize_array(k2)
    #     k3 = normalize_array(k3)
    #     k4 = normalize_array(k4)

    k1 = k1 / 60
    k2 = k2 / 60
    k3 = k3 / 60
    k4 = k4 / 60

    # # 进行最大值最小值归一化
    # if number == 1 :
    #     k1 = normalize_array(k1)
    #     k2 = normalize_array(k2)
    #     k3 = normalize_array(k3)
    #     k4 = normalize_array(k4)

    cp_fdg = np.array(cp_data[0].tolist())
    discriminant = (k2 + k3 + k4) ** 2 - 4 * k2 * k4    #计算 discriminant，用于后续 alpha1 和 alpha2 的计算
    discriminant = np.maximum(discriminant, 0)  # 将负值替换为零
    alpha1 = (k2 + k3 + k4 - np.sqrt(discriminant)) / 2
    alpha2 = (k2 + k3 + k4 + np.sqrt(discriminant)) / 2

    mask = (alpha2 - alpha1) != 0
    # 计算 a
    a = np.zeros_like(k1)
    a[mask] = k1[mask] * (k3[mask] + k4[mask] - alpha1[mask]) / (alpha2[mask] - alpha1[mask])
    # 计算 b
    b = np.zeros_like(k1)
    b[mask] = k1[mask] * (alpha2[mask] - k3[mask] - k4[mask]) / (alpha2[mask] - alpha1[mask])

    # alpha1 = (k2 + k3 + k4 - np.sqrt((k2 + k3 + k4) ** 2 - 4 * k2 * k4)) / 2
    # alpha2 = (k2 + k3 + k4 + np.sqrt((k2 + k3 + k4) ** 2 - 4 * k2 * k4)) / 2
    # a = k1 * (k3 + k4 - alpha1) / (alpha2 - alpha1)  # a: 128*128
    # b = k1 * (alpha2 - k3 - k4) / (alpha2 - alpha1)

    T = len(cp_fdg)
    array = np.arange(1, T + 1)  # array:(3600,)
    array = array.reshape((1, 1, T))  # 1*1*3600
    a = np.repeat(a[:, :, np.newaxis], T, axis=2)  # a: 128*128*3600
    b = np.repeat(b[:, :, np.newaxis], T, axis=2)  # b: 128*128*3600

    alpha1 = np.repeat(alpha1[:, :, np.newaxis], T, axis=2)
    alpha2 = np.repeat(alpha2[:, :, np.newaxis], T, axis=2)
    part11 = a * cp_fdg  # (128*128*3600)   计算部分和频域卷积
    part12 = np.exp(-alpha1 * array)  # (128*128*3600)

    part21 = b * cp_fdg  # (128*128*3600)
    part22 = np.exp(-alpha2 * array)  # (128*128*3600)

    # 新卷积方法
    # CT1 = fftconvolve(part11, part12, mode='full', axes=2)
    # CT2 = fftconvolve(part21, part22, mode='full', axes=2)
    # CT1 = CT1[:, :, :T]
    # CT2 = CT2[:, :, :T]
    #
    # CT = CT1 + CT2
    # 新卷积方法 进行傅里叶变换并计算最终CT
    temp_part11 = np.fft.fft(part11)
    temp_part12 = np.fft.fft(part12)
    CT1_temp = np.fft.ifft(temp_part11 * temp_part12)
    CT1_temp = np.real(CT1_temp)
    CT1 = CT1_temp[:, :, :T]

    temp_part21 = np.fft.fft(part21)
    temp_part22 = np.fft.fft(part22)
    CT2_temp = np.fft.ifft(temp_part21 * temp_part22)
    CT2_temp = np.real(CT2_temp)
    CT2 = CT2_temp[:, :, :T]
    CT = CT1 + CT2

    return CT


def calculate_xm(tms, tme, CPET, lmbda):    #用于计算某个时间段内的积分值
    # print(CPET.shape[2] + 1)
    t_values = np.arange(0, CPET.shape[2])  # 假设 CPET 包含3600个时间点，可以自行调整

    time_indices = np.where((t_values >= tms) & (t_values <= tme))[0]  # 获取在 tms 和 tme 范围内的时间索引
    # print(time_indices)
    CPET_sub = CPET[:, :, time_indices]  # 截取对应时间段的 CPET 数据
    t_sub = t_values[time_indices]  # 对应的时间值
    # print(f"CPET_sub min at time {tms}-{tme}: {np.min(CPET_sub)}")
    # print(f"CPET_sub max at time {tms}-{tme}: {np.max(CPET_sub)}")
    integrand = CPET_sub * np.exp(-lmbda * t_sub)   #计算积分的被积函数
    xm = trapz(integrand, t_sub)    #计算积分
    # 在 calculate_xm 中添加打印语句
    # print("lmbda:", lmbda)  # 确保 lambda_value 正确传递
    # print("CPET_sub shape:", CPET_sub.shape)
    # print("t_sub:", t_sub)

    return xm

if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
