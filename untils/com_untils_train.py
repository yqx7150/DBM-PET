# import os
# import cv2
import numpy as np
import torch
import tensorflow as tf
from scipy.integrate import trapz

import torch.nn.functional as F
from scipy.io import savemat, loadmat
import torch.fft


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


def get_mean_k_data_torch_new_model(reconstruct_for):
    """
    对numpy类型的数据进行解耦,将(1,48,128,128)的数据-->4 x (1,12,128,128)然后进行平均
    Args:
        reconstruct_for:网络的预测K值(1,48,128,128)

    Returns:预测的k1,k2,k3,k4   4 x (128x128)

    """
    # print(reconstruct_for.shape)
    k1, k2, k3, k4 = torch.split(reconstruct_for, 1, dim=1)
    # print(k1.squeeze().shape)
    # print(k1.dtype)
    # for i in range(11):
    #     if torch.equal(k1[:, i, :, :], k1[:, i+1, :, :]):
    #         print("tensor1 and tensor2 are equal.")
    #     else:
    #         print("tensor1 and tensor2 are not equal.")
    # assert 0
    # print(k1.shape)
    # d3_k1 = k1.squeeze()
    # print(d3_k1.shape)
    # d3_k1 = d3_k1.double()
    # pred_k1 = torch.mean(d3_k1, dim=0)
    # # if torch.equal(pred_k1, d3_k1[0, :, :].squeeze()):
    # if torch.equal(pred_k1.float(), k1[:, 0, :, :].squeeze()):
    #     print("tensor1 and tensor2 are equal.")
    # else:
    #     print("tensor1 and tensor2 are not equal.")
    # assert 0

    # pred_k1 = torch.mean(k1.squeeze().double(), dim=0).float()
    # pred_k2 = torch.mean(k2.squeeze(), dim=0).float()
    # pred_k3 = torch.mean(k3.squeeze(), dim=0).float()
    # pred_k4 = torch.mean(k4.squeeze(), dim=0).float()
    pred_k1 = torch.mean(k1.squeeze().double(), dim=0).float()
    pred_k2 = torch.mean(k2.squeeze().double(), dim=0).float()
    pred_k3 = torch.mean(k3.squeeze().double(), dim=0).float()
    pred_k4 = torch.mean(k4.squeeze().double(), dim=0).float()

    # k1 = k1[:, 1, :, :].squeeze()
    # k1_data_old = loadmat('./data/fmz_zubal_head_sample3_kmin_noise_test/train/k1/fmz_k1_1.mat')['data_new']
    # k1_data_old = torch.from_numpy(k1_data_old).float().cuda()
    # # print(torch.max(k1))
    # # print(torch.max(pred_k1))
    # print(pred_k1.shape)
    # print(k1.shape)
    # print(k1_data_old.shape)
    # # print(pred_k1.dtype)
    # # print(k1_data_old.dtype)
    # # print(k1_data_old.device)
    # # print(pred_k1.device)
    # if torch.equal(k1_data_old, pred_k1):
    #     print("tensor1 and tensor2 are equal.")
    # else:
    #     print("tensor1 and tensor2 are not equal.")
    # assert 0

    return pred_k1, pred_k2, pred_k3, pred_k4


def get_mean_k_data_torch(reconstruct_for):
    """
    对numpy类型的数据进行解耦,将(1,48,128,128)的数据-->4 x (1,12,128,128)然后进行平均
    Args:
        reconstruct_for:网络的预测K值(1,48,128,128)

    Returns:预测的k1,k2,k3,k4   4 x (128x128)

    """
    # print(reconstruct_for.shape)
    k1, k2, k3, k4 = torch.split(reconstruct_for, 12, dim=1)
    # print(k1.squeeze().shape)
    # print(k1.dtype)
    # for i in range(11):
    #     if torch.equal(k1[:, i, :, :], k1[:, i+1, :, :]):
    #         print("tensor1 and tensor2 are equal.")
    #     else:
    #         print("tensor1 and tensor2 are not equal.")
    # assert 0
    # print(k1.shape)
    # d3_k1 = k1.squeeze()
    # print(d3_k1.shape)
    # d3_k1 = d3_k1.double()
    # pred_k1 = torch.mean(d3_k1, dim=0)
    # # if torch.equal(pred_k1, d3_k1[0, :, :].squeeze()):
    # if torch.equal(pred_k1.float(), k1[:, 0, :, :].squeeze()):
    #     print("tensor1 and tensor2 are equal.")
    # else:
    #     print("tensor1 and tensor2 are not equal.")
    # assert 0

    # pred_k1 = torch.mean(k1.squeeze().double(), dim=0).float()
    # pred_k2 = torch.mean(k2.squeeze(), dim=0).float()
    # pred_k3 = torch.mean(k3.squeeze(), dim=0).float()
    # pred_k4 = torch.mean(k4.squeeze(), dim=0).float()
    pred_k1 = torch.mean(k1.squeeze().double(), dim=0).float()
    pred_k2 = torch.mean(k2.squeeze().double(), dim=0).float()
    pred_k3 = torch.mean(k3.squeeze().double(), dim=0).float()
    pred_k4 = torch.mean(k4.squeeze().double(), dim=0).float()

    # k1 = k1[:, 1, :, :].squeeze()
    # k1_data_old = loadmat('./data/fmz_zubal_head_sample3_kmin_noise_test/train/k1/fmz_k1_1.mat')['data_new']
    # k1_data_old = torch.from_numpy(k1_data_old).float().cuda()
    # # print(torch.max(k1))
    # # print(torch.max(pred_k1))
    # print(pred_k1.shape)
    # print(k1.shape)
    # print(k1_data_old.shape)
    # # print(pred_k1.dtype)
    # # print(k1_data_old.dtype)
    # # print(k1_data_old.device)
    # # print(pred_k1.device)
    # if torch.equal(k1_data_old, pred_k1):
    #     print("tensor1 and tensor2 are equal.")
    # else:
    #     print("tensor1 and tensor2 are not equal.")
    # assert 0

    return pred_k1, pred_k2, pred_k3, pred_k4


def compare_ms_ssim(pred, target):
    """
    对数据求ms_ssim指标
    Args:
        pred:
        target:

    Returns:

    """
    pred = tf.convert_to_tensor(pred)
    pred = tf.expand_dims(pred, axis=-1)

    target = tf.convert_to_tensor(target)
    target = tf.expand_dims(target, axis=-1)
    pred = tf.cast(pred, dtype=tf.float64)
    target = tf.cast(target, dtype=tf.float64)
    max_val = 1

    result = tf.image.ssim_multiscale(pred, target, max_val, filter_size=1)
    result = result.numpy()
    return result


def compute_mean_loss(reconstruct_for, target_forward_data, mse_loss=False):
    """
    计算每个预测的k值和实际值的loss
    Args:
        reconstruct_for:
        target_forward_data:

    Returns:

    """
    pred_k1, pred_k2, pred_k3, pred_k4 = get_mean_k_data_torch(reconstruct_for)
    k1, k2, k3, k4 = get_mean_k_data_torch(target_forward_data)  # torch.Size([1, 12, 128, 128])
    if mse_loss:
        loss_k1 = F.mse_loss(pred_k1, k1)
        loss_k2 = F.mse_loss(pred_k2, k2)
        loss_k3 = F.mse_loss(pred_k3, k3)
        loss_k4 = F.mse_loss(pred_k4, k4)
    else:
        loss_k1 = F.huber_loss(pred_k1, k1)
        loss_k2 = F.huber_loss(pred_k2, k2)
        loss_k3 = F.huber_loss(pred_k3, k3)
        loss_k4 = F.huber_loss(pred_k4, k4)

    return loss_k1, loss_k2, loss_k3, loss_k4


def compute_mean_loss_new_model(reconstruct_for, target_forward_data, mse_loss=False):
    """
    计算每个预测的k值和实际值的loss
    Args:
        reconstruct_for:
        target_forward_data:

    Returns:

    """
    pred_k1, pred_k2, pred_k3, pred_k4 = get_mean_k_data_torch_new_model(reconstruct_for)
    k1, k2, k3, k4 = get_mean_k_data_torch_new_model(target_forward_data)  # torch.Size([1, 12, 128, 128])
    if mse_loss:
        loss_k1 = F.mse_loss(pred_k1, k1)
        loss_k2 = F.mse_loss(pred_k2, k2)
        loss_k3 = F.mse_loss(pred_k3, k3)
        loss_k4 = F.mse_loss(pred_k4, k4)
    else:
        loss_k1 = F.huber_loss(pred_k1, k1)
        loss_k2 = F.huber_loss(pred_k2, k2)
        loss_k3 = F.huber_loss(pred_k3, k3)
        loss_k4 = F.huber_loss(pred_k4, k4)

    return loss_k1, loss_k2, loss_k3, loss_k4

def compute_mean_loss_rev(reconstruct_for, target_forward_data, mse_loss=False):
    """
    计算每个预测的输入值和实际值的输入之间的loss
    Args:
        reconstruct_for:
        target_forward_data:

    Returns:

    """

    pred_data1, pred_data2, pred_data3, pred_data4 = torch.split(reconstruct_for, 12, dim=1)
    if mse_loss:
        rev_loss1 = F.mse_loss(target_forward_data.squeeze(), pred_data1.squeeze())
        rev_loss2 = F.mse_loss(target_forward_data.squeeze(), pred_data2.squeeze())
        rev_loss3 = F.mse_loss(target_forward_data.squeeze(), pred_data3.squeeze())
        rev_loss4 = F.mse_loss(target_forward_data.squeeze(), pred_data4.squeeze())
    else:
        rev_loss1 = F.huber_loss(target_forward_data.squeeze(), pred_data1.squeeze())
        rev_loss2 = F.huber_loss(target_forward_data.squeeze(), pred_data2.squeeze())
        rev_loss3 = F.huber_loss(target_forward_data.squeeze(), pred_data3.squeeze())
        rev_loss4 = F.huber_loss(target_forward_data.squeeze(), pred_data4.squeeze())

    return rev_loss1, rev_loss2, rev_loss3, rev_loss4


def normalize_tensor_torch(tensor):
    """
    最大最小值归一化 对 (128, 128, 18) 的 PyTorch 张量
    Args:
        tensor: PyTorch 张量

    Returns:
        normalize_tensor: 归一化后的 PyTorch 张量
    """
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)

    if tensor_max == tensor_min:
        # 分母为零，避免除以零的情况
        normalize_tensor = torch.zeros_like(tensor)
    else:
        normalize_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

    return normalize_tensor



def update_tracer_concentration_torch_test(reconstruct_for_k_data, cp_data, number):
    """
    由预测的k1,k2,k3,k4生成预测的TAC曲线
    Args:
        reconstruct_for_k_data:1,48,128,128
        cp_data:
        number:

    Returns:

    """
    # has_nan_or_inf = torch.isnan(reconstruct_for_k_data).any() or torch.isinf(reconstruct_for_k_data).any()
    # print("reconstruct_for_k_data", has_nan_or_inf)
    k1, k2, k3, k4 = get_mean_k_data_torch(reconstruct_for_k_data)  # 128x128
    k4 = torch.zeros_like(k1)
    # print(k1.shape)
    # print(torch.max(k1))
    # print(torch.min(k1))
    # assert 0
    #
    # print(torch.max(reconstruct_for_k_data))
    # print(torch.min(reconstruct_for_k_data))
    # 进行最大值最小值归一化
    if number == 1:
        k1 = normalize_tensor_torch(k1)
        k2 = normalize_tensor_torch(k2)
        k3 = normalize_tensor_torch(k3)
        k4 = normalize_tensor_torch(k4)

    k1 = k1 / 60
    k2 = k2 / 60
    k3 = k3 / 60
    k4 = k4 / 60

    cp_fmz = torch.tensor(cp_data[0].tolist(), dtype=torch.float32)
    cp_fmz = cp_fmz.float().cuda()


    discriminant = (k2 + k3 + k4) ** 2 - 4 * k2 * k4
    # discriminant = discriminant.clamp(min=0)

    discriminant = torch.maximum(discriminant, torch.tensor(0.0).cuda())
    # discriminant = torch.maximum(discriminant, torch.tensor(0.0))  # 将负值替换为零
    alpha1 = (k2 + k3 + k4 - torch.sqrt(discriminant)) / 2
    alpha2 = (k2 + k3 + k4 + torch.sqrt(discriminant)) / 2

    mask = (alpha2 - alpha1) != 0

    # 计算 a
    a = torch.zeros_like(k1)
    a[mask] = k1[mask] * (k3[mask] + k4[mask] - alpha1[mask]) / (alpha2[mask] - alpha1[mask])

    # 计算 b
    b = torch.zeros_like(k1)
    b[mask] = k1[mask] * (alpha2[mask] - k3[mask] - k4[mask]) / (alpha2[mask] - alpha1[mask])

    T = len(cp_fmz)
    array = torch.arange(1, T + 1, dtype=torch.float32)  # array:(3600,)
    array = array.reshape((1, 1, T))  # 1*1*3600
    a = a.unsqueeze(2).repeat(1, 1, T)
    b = b.unsqueeze(2).repeat(1, 1, T)
    # a = a.unsqueeze(2).repeat(1, 1, T).float().cuda()  # a: 128*128*3600
    # b = b.unsqueeze(2).repeat(1, 1, T).float().cuda()  # b: 128*128*3600

    alpha1 = alpha1.unsqueeze(2).repeat(1, 1, T)
    alpha2 = alpha2.unsqueeze(2).repeat(1, 1, T)

    array = array.cuda()
    part11 = a * cp_fmz  # (128*128*3600)
    part12 = torch.exp(-alpha1 * array)  # (128*128*3600)

    part21 = b * cp_fmz  # (128*128*3600)
    part22 = torch.exp(-alpha2 * array)  # (128*128*3600)

    # 新卷积方法
    temp_part11 = torch.fft.fft(part11)
    temp_part12 = torch.fft.fft(part12)
    CT1_temp = torch.fft.ifft(temp_part11 * temp_part12)
    CT1_temp = torch.real(CT1_temp)
    CT1 = CT1_temp[:, :, :T]

    temp_part21 = torch.fft.fft(part21)
    temp_part22 = torch.fft.fft(part22)
    CT2_temp = torch.fft.ifft(temp_part21 * temp_part22)
    CT2_temp = torch.real(CT2_temp)
    CT2 = CT2_temp[:, :, :T]
    CT = CT1 + CT2

    return CT


def update_tracer_concentration_torch(reconstruct_for_k_data, cp_data, number):
    """
    由预测的k1,k2,k3,k4生成预测的TAC曲线
    Args:
        reconstruct_for_k_data:1,48,128,128
        cp_data:
        number:

    Returns:

    """
    # has_nan_or_inf = torch.isnan(reconstruct_for_k_data).any() or torch.isinf(reconstruct_for_k_data).any()
    # print("reconstruct_for_k_data", has_nan_or_inf)
    k1, k2, k3, k4 = get_mean_k_data_torch(reconstruct_for_k_data)  # 128x128
    # print(k1.shape)
    # print(torch.max(k1))
    # print(torch.min(k1))
    # assert 0
    #
    # print(torch.max(reconstruct_for_k_data))
    # print(torch.min(reconstruct_for_k_data))
    # 进行最大值最小值归一化
    if number == 1:
        k1 = normalize_tensor_torch(k1)
        k2 = normalize_tensor_torch(k2)
        k3 = normalize_tensor_torch(k3)
        k4 = normalize_tensor_torch(k4)

    k1 = k1 / 60
    k2 = k2 / 60
    k3 = k3 / 60
    k4 = k4 / 60

    cp_fmz = torch.tensor(cp_data[0].tolist(), dtype=torch.float32)
    cp_fmz = cp_fmz.float().cuda()


    discriminant = (k2 + k3 + k4) ** 2 - 4 * k2 * k4
    # discriminant = discriminant.clamp(min=0)

    discriminant = torch.maximum(discriminant, torch.tensor(0.0).cuda())
    # discriminant = torch.maximum(discriminant, torch.tensor(0.0))  # 将负值替换为零
    alpha1 = (k2 + k3 + k4 - torch.sqrt(discriminant)) / 2
    alpha2 = (k2 + k3 + k4 + torch.sqrt(discriminant)) / 2

    mask = (alpha2 - alpha1) != 0

    # 计算 a
    a = torch.zeros_like(k1)
    a[mask] = k1[mask] * (k3[mask] + k4[mask] - alpha1[mask]) / (alpha2[mask] - alpha1[mask])

    # 计算 b
    b = torch.zeros_like(k1)
    b[mask] = k1[mask] * (alpha2[mask] - k3[mask] - k4[mask]) / (alpha2[mask] - alpha1[mask])

    T = len(cp_fmz)
    array = torch.arange(1, T + 1, dtype=torch.float32)  # array:(3600,)
    array = array.reshape((1, 1, T))  # 1*1*3600
    a = a.unsqueeze(2).repeat(1, 1, T)
    b = b.unsqueeze(2).repeat(1, 1, T)
    # a = a.unsqueeze(2).repeat(1, 1, T).float().cuda()  # a: 128*128*3600
    # b = b.unsqueeze(2).repeat(1, 1, T).float().cuda()  # b: 128*128*3600

    alpha1 = alpha1.unsqueeze(2).repeat(1, 1, T)
    alpha2 = alpha2.unsqueeze(2).repeat(1, 1, T)

    array = array.cuda()
    part11 = a * cp_fmz  # (128*128*3600)
    part12 = torch.exp(-alpha1 * array)  # (128*128*3600)

    part21 = b * cp_fmz  # (128*128*3600)
    part22 = torch.exp(-alpha2 * array)  # (128*128*3600)

    # 新卷积方法
    temp_part11 = torch.fft.fft(part11)
    temp_part12 = torch.fft.fft(part12)
    CT1_temp = torch.fft.ifft(temp_part11 * temp_part12)
    CT1_temp = torch.real(CT1_temp)
    CT1 = CT1_temp[:, :, :T]

    temp_part21 = torch.fft.fft(part21)
    temp_part22 = torch.fft.fft(part22)
    CT2_temp = torch.fft.ifft(temp_part21 * temp_part22)
    CT2_temp = torch.real(CT2_temp)
    CT2 = CT2_temp[:, :, :T]
    CT = CT1 + CT2

    return CT


def calculate_xm_torch(tms, tme, CPET, lmbda):
    # print(CPET.shape[2] + 1)
    t_values = np.arange(0, CPET.shape[2])  # 假设 CPET 包含3600个时间点，可以自行调整

    time_indices = np.where((t_values >= tms) & (t_values <= tme))[0]  # 获取在 tms 和 tme 范围内的时间索引
    time_indices = torch.from_numpy(time_indices)
    t_values = torch.from_numpy(t_values)
    # print(time_indices)

    CPET_sub = CPET[:, :, time_indices]  # 截取对应时间段的 CPET 数据
    t_sub = t_values[time_indices].cuda()  # 对应的时间值
    lmbda = lmbda.cuda()
    # print(lmbda.device)
    # print(t_sub.device)
    # print(CPET_sub.device)
    integrand = CPET_sub * torch.exp(-lmbda * t_sub)
    xm = torch.trapz(integrand, t_sub)

    return xm


def calculate_img_torch(reconstruct_for_k_data, sampling_intervals, cp_data):
    """
    由CP(t)和CT(t)，采样协议，计算18帧的数据
    :param sampling_intervals:
    :param cp_data:
    :param CT:
    :return:
    """
    # reconstruct_for_k_data = reconstruct_for_k_data.cpu().detach().numpy()  # 128,128,12
    CP_FMZ = cp_data
    CT_FMZ = update_tracer_concentration_torch(reconstruct_for_k_data, CP_FMZ, 0)
    f_FMZ = torch.zeros((128, 128, 18)).cuda()
    lambda_value = np.log(2) / (20.4 * 60)
    lambda_value = torch.tensor(lambda_value, dtype=torch.float32)
    start_index = 0
    for k in range(len(sampling_intervals)):
        end_index = start_index + sampling_intervals[k] - 1
        f_FMZ[:, :, k] = calculate_xm_torch(start_index, end_index, CT_FMZ, lambda_value)
        start_index = end_index + 1
    f_FMZ = f_FMZ / torch.max(f_FMZ)
    return f_FMZ


def calculate_img_logan_no_weight(reconstruct_for_k_data, target_forward_k_data, sampling_intervals, cp_data,
                                  ki_data_btach,
                                  vb_data_batch):
    """
        logan_no_weight,loss不加权重
        对后10帧的数据进行计算logan分析损失：y=ax+b方式
        Args:
            reconstruct_for_k_data:网络输出的k1-k4结果
            target_forward_k_data:实际的k1-k4结果
            sampling_intervals:采样协议
            cp_data:血浆数据
            ki_data_btach:logan分析中的斜率
            vb_data_batch:logan分析中的截距

        Returns:预测的和实际的logan分析

        """
    CP_FMZ = torch.from_numpy(cp_data).cuda()

    pred_CT_FMZ = update_tracer_concentration_torch(reconstruct_for_k_data, CP_FMZ, 0)
    target_CT_FMZ = update_tracer_concentration_torch(target_forward_k_data, CP_FMZ, 0)
    CP_FMZ = CP_FMZ.repeat(128, 128, 1)

    x = torch.zeros((128, 128, 18)).cuda()
    y = torch.zeros((128, 128, 18)).cuda()
    base = 0
    for i in range(18):
        index = base + sampling_intervals[i] // 2

        integral_Cp = torch.trapz(CP_FMZ[:, :, :index + 1], dim=2)
        integral_CT = torch.trapz(pred_CT_FMZ[:, :, :index + 1], dim=2)
        CT_values_target = target_CT_FMZ[:, :, index]
        CT_values_pred = pred_CT_FMZ[:, :, index]
        # x_roi = torch.where(CT_values_target != 0, torch.div(integral_Cp, CT_values_target),
        #                     torch.zeros_like(integral_Cp))
        # y_roi = torch.where(CT_values_pred != 0, torch.div(integral_CT, CT_values_pred), torch.zeros_like(integral_CT))
        # 创建掩码
        mask_x = CT_values_target != 0
        mask_y = CT_values_pred != 0

        # 使用掩码进行除法
        x_roi = integral_Cp / torch.where(mask_x, CT_values_target, torch.ones_like(CT_values_target))
        y_roi = integral_CT / torch.where(mask_y, CT_values_pred, torch.ones_like(CT_values_pred))
        # x_roi = torch.where(mask_x, torch.div(integral_Cp, CT_values_target), torch.zeros_like(integral_Cp))
        # y_roi = torch.where(mask_y, torch.div(integral_CT, CT_values_pred), torch.zeros_like(integral_CT))

        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]
    x = x[:, :, 12:18]
    y = y[:, :, 12:18]

    ki_data_btach = ki_data_btach.permute(1, 2, 0).to(dtype=torch.float32).cuda()
    vb_data_batch = vb_data_batch.permute(1, 2, 0).to(dtype=torch.float32).cuda()
    # loss计算1
    target = x * ki_data_btach + vb_data_batch
    pred = y
    # loss计算2
    # diff_x = torch.zeros((128, 128, 5)).cuda()
    # diff_y = torch.zeros((128, 128, 5)).cuda()
    # for i in range(0, 5):
    #     diff_y[:, :, i] = y[:, :, i+1] - y[:, :, i]
    #     diff_x[:, :, i] = x[:, :, i+1] - x[:, :, i]
    #
    # target = ki_data_btach * diff_x
    # pred = diff_y pred = pred.float()
    #     target = target.float()



    return pred, target


def calculate_img_logan2_no_weight(reconstruct_for_k_data, target_forward_k_data, sampling_intervals, cp_data,
                                   ki_data_btach,
                                   vb_data_batch):
    """
    logan2_no_weight
    对后10帧的数据进行计算logan分析损失：做差的方法
    Args:
        reconstruct_for_k_data:网络输出的k1-k4结果
        target_forward_k_data:实际的k1-k4结果
        sampling_intervals:采样协议
        cp_data:血浆数据
        ki_data_btach:logan分析中的斜率
        vb_data_batch:logan分析中的截距

    Returns:预测的和实际的logan分析

    """

    CP_FMZ = torch.from_numpy(cp_data).cuda()

    # target_forward_k_data = torch.from_numpy(target_forward_k_data)
    # reconstruct_for_k_data = torch.abs(reconstruct_for_k_data)
    pred_CT_FMZ = update_tracer_concentration_torch(reconstruct_for_k_data, CP_FMZ, 0)
    target_CT_FMZ = update_tracer_concentration_torch(target_forward_k_data, CP_FMZ, 0)
    # pred_CT_FMZ = target_CT_FMZ
    CP_FMZ = CP_FMZ.repeat(128, 128, 1)

    x = torch.zeros((128, 128, 18)).cuda()
    y = torch.zeros((128, 128, 18)).cuda()
    base = 0
    for i in range(18):
        index = base + sampling_intervals[i] // 2

        integral_Cp = torch.trapz(CP_FMZ[:, :, :index + 1], dim=2)
        integral_CT = torch.trapz(pred_CT_FMZ[:, :, :index + 1], dim=2)
        CT_values_target = target_CT_FMZ[:, :, index]
        CT_values_pred = pred_CT_FMZ[:, :, index]
        mask_x = CT_values_target != 0
        mask_y = CT_values_pred != 0

        # 使用掩码进行除法
        x_roi = integral_Cp / torch.where(mask_x, CT_values_target, torch.ones_like(CT_values_target))
        y_roi = integral_CT / torch.where(mask_y, CT_values_pred, torch.ones_like(CT_values_pred))
        # x_roi = torch.where(CT_values_target != 0, torch.div(integral_Cp, CT_values_target),
        #                     torch.zeros_like(integral_Cp))
        # y_roi = torch.where(CT_values_pred != 0, torch.div(integral_CT, CT_values_pred), torch.zeros_like(integral_CT))

        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]
    x = x[:, :, 8:18]
    y = y[:, :, 8:18]

    ki_data_btach = ki_data_btach.permute(1, 2, 0).to(dtype=torch.float32).cuda()
    vb_data_batch = vb_data_batch.permute(1, 2, 0).to(dtype=torch.float32).cuda()
    # loss计算1
    # target = x * ki_data_btach + vb_data_batch
    # pred = y
    # loss计算2
    diff_x = torch.zeros((128, 128, 9)).cuda()
    diff_y = torch.zeros((128, 128, 9)).cuda()
    for i in range(0, 9):
        diff_y[:, :, i] = y[:, :, i + 1] - y[:, :, i]
        diff_x[:, :, i] = x[:, :, i + 1] - x[:, :, i]

    target = ki_data_btach * diff_x
    pred = diff_y

    # loss = F.huber_loss(pred, target, reduction='mean')
    # print("no_nor_huber_loss", loss)
    # loss = F.huber_loss(pred1, target1, reduction='mean')
    # print("no_nor_huber_loss1", loss)
    # pred = pred / torch.max(pred)
    pred = pred.float()
    # target = target / torch.max(target)
    target = target.float()

    return pred, target


def calculate_img_logan3_no_weight(reconstruct_for_k_data, target_forward_k_data, sampling_intervals, cp_data):
    """
        logan_no_weight,loss不加权重
        对后10帧的数据进行计算logan分析损失：y=ax+b方式
        Args:
            reconstruct_for_k_data:网络输出的k1-k4结果
            target_forward_k_data:实际的k1-k4结果
            sampling_intervals:采样协议
            cp_data:血浆数据
            ki_data_btach:logan分析中的斜率
            vb_data_batch:logan分析中的截距

        Returns:预测的和实际的logan分析

        """
    CP_FMZ = torch.from_numpy(cp_data).cuda()

    pred_CT_FMZ = update_tracer_concentration_torch(reconstruct_for_k_data, CP_FMZ, 0)
    target_CT_FMZ = update_tracer_concentration_torch(target_forward_k_data, CP_FMZ, 0)
    # print(torch.max(pred_CT_FMZ))
    # print(torch.min(pred_CT_FMZ))
    # print(torch.max(target_CT_FMZ))
    # print(torch.min(target_CT_FMZ))
    # assert 0
    CP_FMZ = CP_FMZ.repeat(128, 128, 1)

    x = torch.zeros((128, 128, 18)).cuda()
    y = torch.zeros((128, 128, 18)).cuda()
    base = 0
    for i in range(18):
        index = base + sampling_intervals[i] // 2

        integral_Cp = torch.trapz(CP_FMZ[:, :, :index + 1], dim=2)
        integral_CT = torch.trapz(pred_CT_FMZ[:, :, :index + 1], dim=2)
        # integral_target_CT = torch.trapz(target_CT_FMZ[:, :, :index + 1], dim=2)
        CT_values_target = target_CT_FMZ[:, :, index]
        CT_values_pred = pred_CT_FMZ[:, :, index]
        # x_roi = torch.where(CT_values_target != 0, torch.div(integral_Cp, CT_values_target),
        #                     torch.zeros_like(integral_Cp))
        # y_roi = torch.where(CT_values_pred != 0, torch.div(integral_CT, CT_values_pred), torch.zeros_like(integral_CT))
        # 创建掩码
        mask_x = CT_values_target != 0
        mask_y = CT_values_pred != 0
        # 使用掩码进行除法
        x_roi = integral_Cp / torch.where(mask_x, CT_values_target, torch.ones_like(CT_values_target))
        y_roi = integral_CT / torch.where(mask_y, CT_values_pred, torch.ones_like(CT_values_pred))
        # 测试代码
        # x_roi = integral_Cp / torch.where(mask_x, CT_values_target, torch.ones_like(CT_values_target))
        # y_roi = integral_target_CT / torch.where(mask_x, CT_values_target, torch.ones_like(CT_values_target))

        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]

    nihe_x = x[:, :, 8:18]
    nihe_y = y[:, :, 8:18]
    #
    # print(torch.max(nihe_x))
    # print(torch.min(nihe_x))
    # print(torch.max(nihe_y))
    # print(torch.min(nihe_y))
    # assert 0
    rows, columns, time_points = x.shape
    K_values = torch.zeros((rows, columns)).cuda()
    b_values = torch.zeros((rows, columns)).cuda()

    for i in range(rows):
        for j in range(columns):
            # Get x and y at the corresponding position
            x_flat = nihe_x[i, j, :].view(-1)
            y_flat = nihe_y[i, j, :].view(-1)

            # Check if all elements are 0, if so, skip
            if torch.all(x_flat == 0) or torch.all(y_flat == 0):
                continue

            # Augment the input with a column of ones for the bias term
            X_augmented = torch.stack([x_flat, torch.ones_like(x_flat)], dim=1)
            # print(X_augmented.shape)
            # print(y_flat.shape)
            # has_nan_or_inf = torch.logical_or(torch.isnan(y_flat), torch.isinf(y_flat)).any().item()
            # print("pred_ki是否存在 NaN 或 Inf:", has_nan_or_inf)
            # assert 0
            # Use torch.linalg.lstsq to solve for coefficients
            coefficients = torch.linalg.lstsq(X_augmented, y_flat.view(-1, 1)).solution

            # Extract slope (K) and intercept (b)
            K_values[i, j] = coefficients[0].item()
            b_values[i, j] = coefficients[1].item()

    return K_values, b_values


def calculate_img_logan3_no_weight_test(reconstruct_for_k_data, target_forward_k_data, sampling_intervals, cp_data):
    """
        logan_no_weight,loss不加权重
        对后10帧的数据进行计算logan分析损失：y=ax+b方式
        Args:
            reconstruct_for_k_data:网络输出的k1-k4结果
            target_forward_k_data:实际的k1-k4结果
            sampling_intervals:采样协议
            cp_data:血浆数据
            ki_data_btach:logan分析中的斜率
            vb_data_batch:logan分析中的截距

        Returns:预测的和实际的logan分析

        """
    CP_FMZ = torch.from_numpy(cp_data).cuda()

    # pred_CT_FMZ = update_tracer_concentration_torch(reconstruct_for_k_data, CP_FMZ, 0)
    target_CT_FMZ = update_tracer_concentration_torch(target_forward_k_data, CP_FMZ, 0)
    # print(torch.max(pred_CT_FMZ))
    # print(torch.min(pred_CT_FMZ))
    # print(torch.max(target_CT_FMZ))
    # print(torch.min(target_CT_FMZ))
    # assert 0

    CP_FMZ = CP_FMZ.repeat(128, 128, 1)

    x = torch.zeros((128, 128, 18)).cuda()
    y = torch.zeros((128, 128, 18)).cuda()
    base = 0
    for i in range(18):
        index = base + sampling_intervals[i] // 2

        integral_Cp = torch.trapz(CP_FMZ[:, :, :index + 1], dim=2)
        # integral_CT = torch.trapz(pred_CT_FMZ[:, :, :index + 1], dim=2)
        integral_target_CT = torch.trapz(target_CT_FMZ[:, :, :index + 1], dim=2)
        CT_values_target = target_CT_FMZ[:, :, index]
        # CT_values_pred = pred_CT_FMZ[:, :, index]
        # x_roi = torch.where(CT_values_target != 0, torch.div(integral_Cp, CT_values_target),
        #                     torch.zeros_like(integral_Cp))
        # y_roi = torch.where(CT_values_pred != 0, torch.div(integral_CT, CT_values_pred), torch.zeros_like(integral_CT))
        # 创建掩码
        mask_x = CT_values_target != 0
        # mask_y = CT_values_pred != 0
        # 使用掩码进行除法
        # x_roi = integral_Cp / torch.where(mask_x, CT_values_target, torch.ones_like(CT_values_target))
        # y_roi = integral_CT / torch.where(mask_y, CT_values_pred, torch.ones_like(CT_values_pred))
        # 测试代码
        x_roi = integral_Cp / torch.where(mask_x, CT_values_target, torch.ones_like(CT_values_target))
        y_roi = integral_target_CT / torch.where(mask_x, CT_values_target, torch.ones_like(CT_values_target))

        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]

    nihe_x = x[:, :, 8:18]
    nihe_y = y[:, :, 8:18]
    #

    rows, columns, time_points = x.shape
    K_values = torch.zeros((rows, columns)).cuda()
    b_values = torch.zeros((rows, columns)).cuda()

    for i in range(rows):
        for j in range(columns):
            # Get x and y at the corresponding position
            x_flat = nihe_x[i, j, :].view(-1)
            y_flat = nihe_y[i, j, :].view(-1)

            # Check if all elements are 0, if so, skip
            if torch.all(x_flat == 0) or torch.all(y_flat == 0):
                continue

            # Augment the input with a column of ones for the bias term
            X_augmented = torch.stack([x_flat, torch.ones_like(x_flat)], dim=1)
            # print(X_augmented.shape)
            # print(y_flat.shape)
            # has_nan_or_inf = torch.logical_or(torch.isnan(y_flat), torch.isinf(y_flat)).any().item()
            # print("pred_ki是否存在 NaN 或 Inf:", has_nan_or_inf)
            # assert 0
            # Use torch.linalg.lstsq to solve for coefficients
            coefficients = torch.linalg.lstsq(X_augmented, y_flat.view(-1, 1)).solution

            # Extract slope (K) and intercept (b)
            K_values[i, j] = coefficients[0].item()
            b_values[i, j] = coefficients[1].item()

    return K_values, b_values


def calculate_img_logan3_no_weight_kb(reconstruct_for_k_data, target_forward_k_data, sampling_intervals, cp_data):
    """
        logan_no_weight,loss不加权重
        对后10帧的数据进行计算logan分析损失：y=ax+b方式
        Args:
            reconstruct_for_k_data:网络输出的k1-k4结果
            target_forward_k_data:实际的k1-k4结果
            sampling_intervals:采样协议
            cp_data:血浆数据
            ki_data_btach:logan分析中的斜率
            vb_data_batch:logan分析中的截距

        Returns:预测的和实际的logan分析

        """
    CP_FMZ = torch.from_numpy(cp_data).cuda()

    pred_CT_FMZ = update_tracer_concentration_torch_test(reconstruct_for_k_data, CP_FMZ, 0)
    target_CT_FMZ = update_tracer_concentration_torch_test(target_forward_k_data, CP_FMZ, 0)
    # print(torch.max(pred_CT_FMZ))
    # print(torch.min(pred_CT_FMZ))
    # print(torch.max(target_CT_FMZ))
    # print(torch.min(target_CT_FMZ))
    # assert 0
    CP_FMZ = CP_FMZ.repeat(128, 128, 1)

    x = torch.zeros((128, 128, 18)).cuda()
    y = torch.zeros((128, 128, 18)).cuda()
    base = 0
    for i in range(18):
        index = base + sampling_intervals[i] // 2

        integral_Cp = torch.trapz(CP_FMZ[:, :, :index + 1], dim=2)
        integral_CT = torch.trapz(pred_CT_FMZ[:, :, :index + 1], dim=2)
        # integral_target_CT = torch.trapz(target_CT_FMZ[:, :, :index + 1], dim=2)
        CT_values_target = target_CT_FMZ[:, :, index]
        CT_values_pred = pred_CT_FMZ[:, :, index]
        # x_roi = torch.where(CT_values_target != 0, torch.div(integral_Cp, CT_values_target),
        #                     torch.zeros_like(integral_Cp))
        # y_roi = torch.where(CT_values_pred != 0, torch.div(integral_CT, CT_values_pred), torch.zeros_like(integral_CT))
        # 创建掩码
        mask_x = CT_values_target != 0
        mask_y = CT_values_pred != 0
        # 使用掩码进行除法
        x_roi = integral_Cp / torch.where(mask_x, CT_values_target, torch.ones_like(CT_values_target))
        y_roi = integral_CT / torch.where(mask_y, CT_values_pred, torch.ones_like(CT_values_pred))
        # 测试代码
        # x_roi = integral_Cp / torch.where(mask_x, CT_values_target, torch.ones_like(CT_values_target))
        # y_roi = integral_target_CT / torch.where(mask_x, CT_values_target, torch.ones_like(CT_values_target))

        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]

    nihe_x = x[:, :, 8:18]
    nihe_y = y[:, :, 8:18]
    #
    # print(torch.max(nihe_x))
    # print(torch.min(nihe_x))
    # print(torch.max(nihe_y))
    # print(torch.min(nihe_y))
    # assert 0
    rows, columns, time_points = x.shape
    K_values = torch.zeros((rows, columns)).cuda()
    b_values = torch.zeros((rows, columns)).cuda()

    for i in range(rows):
        for j in range(columns):
            # Get x and y at the corresponding position
            x_flat = nihe_x[i, j, :].view(-1)
            y_flat = nihe_y[i, j, :].view(-1)

            # Check if all elements are 0, if so, skip
            if torch.all(x_flat == 0) or torch.all(y_flat == 0):
                continue

            # Augment the input with a column of ones for the bias term
            X_augmented = torch.stack([x_flat, torch.ones_like(x_flat)], dim=1)
            # print(X_augmented.shape)
            # print(y_flat.shape)
            # has_nan_or_inf = torch.logical_or(torch.isnan(y_flat), torch.isinf(y_flat)).any().item()
            # print("pred_ki是否存在 NaN 或 Inf:", has_nan_or_inf)
            # assert 0
            # Use torch.linalg.lstsq to solve for coefficients
            coefficients = torch.linalg.lstsq(X_augmented, y_flat.view(-1, 1)).solution

            # Extract slope (K) and intercept (b)
            K_values[i, j] = coefficients[0].item()
            b_values[i, j] = coefficients[1].item()

    return K_values, b_values