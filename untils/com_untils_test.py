import os
import cv2
import numpy as np
import openpyxl
import torch
import tensorflow as tf
from scipy import io
from scipy.integrate import cumtrapz, trapz
from scipy.signal import fftconvolve
import torch.nn.functional as F
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import structural_similarity as compare_ssim
from openpyxl import Workbook


def get_mean_k_data_np(reconstruct_for):
    k1, k2, k3, k4 = np.split(reconstruct_for, 4, axis=1)
    pred_k1 = np.mean(k1.squeeze(), axis=0)
    pred_k2 = np.mean(k2.squeeze(), axis=0)
    pred_k3 = np.mean(k3.squeeze(), axis=0)
    pred_k4 = np.mean(k4.squeeze(), axis=0)
    return pred_k1, pred_k2, pred_k3, pred_k4


def process_and_save_metrics(pred_data, target_data, sheet, show_name, save_path, excel_save_path, i_batch, workbook):
    PSNR_FMZ = []
    SSIM_FMZ = []
    MS_SSIM_FMZ = []
    MSE_FMZ = []
    NRMSE_FMZ = []
    # print(i_batch)
    # assert 0
    num = target_data.shape[-1]
    for i in range(num):
        psnr_fmz = compare_psnr(255 * np.abs(pred_data[:, :, i]), 255 * np.abs(target_data[:, :, i]), data_range=255)
        ssim_fmz = compare_ssim(np.abs(target_data[:, :, i]), np.abs(pred_data[:, :, i]), data_range=1)
        mse_fmz = compare_mse(np.abs(target_data[:, :, i]), np.abs(pred_data[:, :, i]))
        ms_ssim_fmz = compare_ms_ssim(np.abs(target_data[:, :, i]), np.abs(pred_data[:, :, i]))

        if not np.all(target_data[:, :, i] == 0):
            nrmse_fmz = compare_nrmse(np.abs(target_data[:, :, i]), np.abs(pred_data[:, :, i]))
        else:
            nrmse_fmz = 0

        PSNR_FMZ.append(psnr_fmz)
        SSIM_FMZ.append(ssim_fmz)
        MS_SSIM_FMZ.append(ms_ssim_fmz)
        MSE_FMZ.append(mse_fmz)
        NRMSE_FMZ.append(nrmse_fmz)

        os.makedirs(save_path + '/pred_data_' + show_name + '_img', exist_ok=True)
        os.makedirs(save_path + '/target_data_' + show_name + '_img', exist_ok=True)
        os.makedirs(save_path + '/pred_data_' + show_name + '_mat', exist_ok=True)
        os.makedirs(save_path + '/target_data_' + show_name + '_mat', exist_ok=True)

        save_img(target_data[:, :, i],
                 save_path+'/target_data_' + show_name + '_img/target_fmz_'+str(i_batch + 1)+'_'+str(i + 1)+'.png')
        save_img(pred_data[:, :, i], save_path+'/pred_data_' + show_name + '_img/pred_fmz_'+str(i_batch + 1)+'_'+str(i + 1)+'.png')

        io.savemat(save_path+'/target_data_' + show_name + '_mat/target_fmz_'+str(i_batch + 1)+'_'+str(i + 1)+'.mat',
                   {'data': target_data[:, :, i]})
        io.savemat(save_path+'/pred_data_' + show_name + '_mat/pred_fmz_'+str(i_batch + 1)+'_'+str(i + 1)+'.mat',
                   {'data': pred_data[:, :, i]})

        data = ['fmz_'+str(i_batch + 1)+'_'+str(i + 1), round(psnr_fmz, 4), round(ssim_fmz, 4),
                round(ms_ssim_fmz, 4), round(mse_fmz, 4), round(nrmse_fmz, 4)]

        sheet.append(data)
        workbook.save(excel_save_path)
    return PSNR_FMZ, SSIM_FMZ, MS_SSIM_FMZ, MSE_FMZ, NRMSE_FMZ



def generate_k_and_b(reconstruct_for_k_data, sampling_intervals, CP_FMZ):
    # 计算积分
    reconstruct_for_k_data = reconstruct_for_k_data.cpu().numpy()
    CT_values = update_tracer_concentration_np(reconstruct_for_k_data, CP_FMZ, 0)
    Cp_values = CP_FMZ
    # CT_values = CT_values.numpy()
    sample_all_time = np.sum(sampling_intervals)
    Cp_values = np.tile(Cp_values, (128, 128, 1))
    sample_time = np.arange(0, sample_all_time, 1)
    Cp_values = Cp_values[:, :, :3600]
    sample_time = sample_time[:3600]
    CT_values = CT_values[:, :, :3600]
    integral_Cp = cumtrapz(Cp_values, sample_time, axis=2, initial=0)
    integral_CT = cumtrapz(CT_values, sample_time, axis=2, initial=0)
    x_roi = np.divide(integral_Cp, CT_values, out=np.zeros_like(integral_Cp), where=CT_values != 0)
    y_roi = np.divide(integral_CT, CT_values, out=np.zeros_like(integral_CT), where=CT_values != 0)
    x = x_roi[:, :, 600:3600]  # 假设 x 的范围为 602 到 3601
    y = y_roi[:, :, 600:3600]  # 假设 y 的范围为 602 到 3601

    rows, columns, time_points = x.shape

    # 初始化存储 K 和 b 的数组
    K_values = np.zeros((rows, columns))
    b_values = np.zeros((rows, columns))

    for i in range(rows):
        for j in range(columns):
            # 获取对应位置的 x 和 y
            x_flat = x[i, j, :].reshape(-1)
            y_flat = y[i, j, :].reshape(-1)

            # 检查是否全为0，如果是则跳过
            if np.all(x_flat == 0) or np.all(y_flat == 0):
                continue

            # 进行线性拟合

            linear_fit = np.polyfit(x_flat, y_flat, 1)

            # 提取斜率 K 和截距 b
            K_values[i, j] = linear_fit[0]
            b_values[i, j] = linear_fit[1]

    return K_values, b_values

def generate_k_and_b_logan(reconstruct_for_k_data, sampling_intervals, CP_FMZ):
    # 计算积分
    reconstruct_for_k_data = reconstruct_for_k_data.cpu().numpy()
    CT_values = update_tracer_concentration_np(reconstruct_for_k_data, CP_FMZ, 0)
    Cp_values = np.array(CP_FMZ[0].tolist())
    # Cp_values = CP_FMZ
    # CT_values = np.array(CT_values)
    # CT_values = CT_values.numpy()
    # print(CT_values.shape)
    # assert 0
    sample_all_time = np.sum(sampling_intervals)
    Cp_values = np.tile(Cp_values, (128, 128, 1))

    x = np.zeros((128, 128, 18))
    y = np.zeros((128, 128, 18))
    base = 0
    for i in range(0, 18):
        # print("i=", i)
        index = base + sampling_intervals[i] // 2
        index_intergral = base + sampling_intervals[i]
        # print("index=", index)
        # print("index_intergral=", index_intergral)
        integral_Cp = np.trapz(Cp_values[:, :, :index + 1], axis=2)
        integral_CT = np.trapz(CT_values[:, :, :index + 1], axis=2)
        # integral_Cp = np.trapz(Cp_values[:, :, :index_intergral + 1], axis=2)
        # integral_CT = np.trapz(CT_values[:, :, :index_intergral + 1], axis=2)
        CT_values1 = CT_values[:, :, index]
        x_roi = np.divide(integral_Cp, CT_values1, out=np.zeros_like(integral_Cp), where=CT_values1 != 0)
        y_roi = np.divide(integral_CT, CT_values1, out=np.zeros_like(integral_CT), where=CT_values1 != 0)
        x[:, :, i] = x_roi
        y[:, :, i] = y_roi
        base = base + sampling_intervals[i]

    # x = x_roi[:, :, 600:3600]  # 假设 x 的范围为 602 到 3601
    # y = y_roi[:, :, 600:3600]  # 假设 y 的范围为 602 到 3601

    rows, columns, time_points = x.shape

    # 初始化存储 K 和 b 的数组
    K_values = np.zeros((rows, columns))
    b_values = np.zeros((rows, columns))
    nihe_x = x[:, :, 8:18]
    nihe_y = y[:, :, 8:18]
    for i in range(rows):
        for j in range(columns):
            # 获取对应位置的 x 和 y
            x_flat = nihe_x[i, j, :].reshape(-1)
            y_flat = nihe_y[i, j, :].reshape(-1)

            # 检查是否全为0，如果是则跳过
            if np.all(x_flat == 0) or np.all(y_flat == 0):
                continue

            # 进行线性拟合
            linear_fit = np.polyfit(x_flat, y_flat, 1)

            # 提取斜率 K 和截距 b
            K_values[i, j] = linear_fit[0]
            b_values[i, j] = linear_fit[1]

    return K_values, b_values

def compare_psnr_show_save(PSNR, SSIM, MSE, MS_SSIM, NRMSE, show_name, save_path, index, ckpt_allname):
    ave_psnr = sum(PSNR) / len(PSNR)
    PSNR_std = np.std(PSNR)

    ave_ssim = sum(SSIM) / len(SSIM)
    SSIM_std = np.std(SSIM)

    ave_ms_ssim = sum(MS_SSIM) / len(MS_SSIM)
    MS_SSIM_std = np.std(MS_SSIM)

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
    print('ave_ms_ssim_' + show_name, ave_ms_ssim)
    print('ave_mse_' + show_name, ave_mse)
    print('ave_nrmse_' + show_name, ave_nrmse)

    file_path = os.path.join(save_path, file_name)
    with open(file_path, 'a+') as f:
        f.write('\n' * 3)
        f.write(ckpt_allname + '_' + show_name + '\n')

        f.write('ave_psnr:' + str(ave_psnr) + ' ' * 3 + 'PSNR_std:' + str(PSNR_std) + '\n')

        f.write('ave_ssim:' + str(ave_ssim) + ' ' * 3 + 'SSIM_std:' + str(SSIM_std) + '\n')

        f.write('ave_ms_ssim:' + str(ave_ms_ssim) + ' ' * 3 + 'SSIM_std:' + str(MS_SSIM_std) + '\n')

        f.write('ave_mse:' + str(ave_mse) + ' ' * 3 + 'MSE_std:' + str(MSE_std) + '\n')

        f.write('ave_nrmse:' + str(ave_nrmse) + ' ' * 3 + 'nrmse_std:' + str(NRMSE_std) + '\n')


def save_img_color(img, img_path):
    """

    Args:
        img:
        img_path:

    Returns:

    """
    img = np.clip(img * 255, 0, 255)

    img_1 = img[:, :,
            :: -1]  # img[:,:,::-1]也就是我们任意不改变width维的方式，也不改变height维的方式，仅仅改变channel维的方式，并且是倒序排列，原本的bgr排列方式经过倒序就变成了rgb的通道排列方式。拓展：如果img[::-1, :, :]其实是对图片进行上下翻转， img[:,::-1,:]是对图像进行左右翻转
    cv2.imwrite(img_path, img_1)


def save_img(img, img_path):
    """
    保存img为白底
    Args:
        img:
        img_path:

    Returns:

    """
    img = np.clip(img * 255, 0,
                  255)  # np.clip(a,a_min,a_max,out=None)是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值。该函数的作用是将数组a中的所有数限定到范围a_min和a_max中。a：输入矩阵；a_min：被限定的最小值，所有比a_min小的数都会强制变为a_min；a_max：被限定的最大值，所有比a_max大的数都会强制变为a_max；out：可以指定输出矩阵的对象，shape与a相同
    img = 255 - img  # 保存为白色为底色的图片

    cv2.imwrite(img_path, img)



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


def get_mean_k_data_torch(reconstruct_for):
    """
    对numpy类型的数据进行解耦,将(1,48,128,128)的数据-->4 x (1,12,128,128)然后进行平均
    Args:
        reconstruct_for:网络的预测K值(1,48,128,128)

    Returns:预测的k1,k2,k3,k4   4 x (128x128)

    """
    k1, k2, k3, k4 = torch.split(reconstruct_for, 12, dim=1)
    pred_k1 = torch.mean(k1.squeeze(), dim=0)
    pred_k2 = torch.mean(k2.squeeze(), dim=0)
    pred_k3 = torch.mean(k3.squeeze(), dim=0)
    pred_k4 = torch.mean(k4.squeeze(), dim=0)

    return pred_k1, pred_k2, pred_k3, pred_k4


def get_mean_k_data(reconstruct_for):
    """
    # get_mean_k_data_np的tensor版本
    Args:
        reconstruct_for:

    Returns:

    """
    k1, k2, k3, k4 = torch.split(reconstruct_for, 12, dim=1)
    pred_k1 = torch.mean(k1.squeeze(), dim=0)
    pred_k2 = torch.mean(k2.squeeze(), dim=0)
    pred_k3 = torch.mean(k3.squeeze(), dim=0)
    pred_k4 = torch.mean(k4.squeeze(), dim=0)
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


def compute_mean_loss(reconstruct_for, target_forward_data):
    """
    计算每个预测的k值和实际值的loss
    Args:
        reconstruct_for:
        target_forward_data:

    Returns:

    """
    pred_k1, pred_k2, pred_k3, pred_k4 = get_mean_k_data(reconstruct_for)
    k1, k2, k3, k4 = get_mean_k_data(target_forward_data)  # torch.Size([1, 12, 128, 128])
    loss_k1 = F.mse_loss(pred_k1, k1)
    loss_k2 = F.mse_loss(pred_k2, k2)
    loss_k3 = F.mse_loss(pred_k3, k3)
    loss_k4 = F.mse_loss(pred_k4, k4)
    return loss_k1, loss_k2, loss_k3, loss_k4


def compute_mean_loss_rev(reconstruct_for, target_forward_data):
    """
    计算每个预测的输入值和实际值的输入之间的loss
    Args:
        reconstruct_for:
        target_forward_data:

    Returns:

    """

    pred_data1, pred_data2, pred_data3, pred_data4 = torch.split(reconstruct_for, 12, dim=1)

    rev_loss1 = F.mse_loss(target_forward_data.squeeze(), pred_data1.squeeze())
    rev_loss2 = F.mse_loss(target_forward_data.squeeze(), pred_data2.squeeze())
    rev_loss3 = F.mse_loss(target_forward_data.squeeze(), pred_data3.squeeze())
    rev_loss4 = F.mse_loss(target_forward_data.squeeze(), pred_data4.squeeze())
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


def update_tracer_concentration_np(reconstruct_for_k_data, cp_data, number):
    """
    由预测的k1,k2,k3,k4生成预测的TAC曲线
    Args:
        reconstruct_for_k_data:
        cp_data:
        number:

    Returns:

    """
    k1, k2, k3, k4 = get_mean_k_data_np(reconstruct_for_k_data)  # 128x128
    #

    # 进行最大值最小值归一化
    if number == 1:
        k1 = normalize_array(k1)
        k2 = normalize_array(k2)
        k3 = normalize_array(k3)
        k4 = normalize_array(k4)

    k1 = k1 / 60
    k2 = k2 / 60
    k3 = k3 / 60
    k4 = k4 / 60

    cp_fmz = np.array(cp_data[0].tolist())
    discriminant = (k2 + k3 + k4) ** 2 - 4 * k2 * k4
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

    T = len(cp_fmz)
    array = np.arange(1, T + 1)  # array:(3600,)
    array = array.reshape((1, 1, T))  # 1*1*3600
    a = np.repeat(a[:, :, np.newaxis], T, axis=2)  # a: 128*128*3600
    b = np.repeat(b[:, :, np.newaxis], T, axis=2)  # b: 128*128*3600

    alpha1 = np.repeat(alpha1[:, :, np.newaxis], T, axis=2)
    alpha2 = np.repeat(alpha2[:, :, np.newaxis], T, axis=2)
    part11 = a * cp_fmz  # (128*128*3600)
    part12 = np.exp(-alpha1 * array)  # (128*128*3600)

    part21 = b * cp_fmz  # (128*128*3600)
    part22 = np.exp(-alpha2 * array)  # (128*128*3600)

    # 新卷积方法
    # CT1 = fftconvolve(part11, part12, mode='full', axes=2)
    # CT2 = fftconvolve(part21, part22, mode='full', axes=2)
    # CT1 = CT1[:, :, :T]
    # CT2 = CT2[:, :, :T]

    # CT = CT1 + CT2
    # 新卷积方法
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


def calculate_xm(tms, tme, CPET, lmbda):
    # print(CPET.shape[2] + 1)
    t_values = np.arange(0, CPET.shape[2])  # 假设 CPET 包含3600个时间点，可以自行调整

    time_indices = np.where((t_values >= tms) & (t_values <= tme))[0]  # 获取在 tms 和 tme 范围内的时间索引
    # print(time_indices)
    CPET_sub = CPET[:, :, time_indices]  # 截取对应时间段的 CPET 数据
    t_sub = t_values[time_indices]  # 对应的时间值

    integrand = CPET_sub * np.exp(-lmbda * t_sub)
    xm = trapz(integrand, t_sub)

    return xm


def calculate_img_np(reconstruct_for_k_data, sampling_intervals, cp_data):
    """

    Args:
        reconstruct_for_k_data: 网络预测的结果
        sampling_intervals: 采样协议
        cp_data:血浆

    Returns:预测的18帧图像

    """
    # 由预测的k1-k4 图像和已知的Cp数据生成预测的18帧的数据
    reconstruct_for_k_data = reconstruct_for_k_data.cpu().detach().numpy()  # 128,128,12

    CP_FMZ = cp_data
    pred_CT_FMZ = update_tracer_concentration_np(reconstruct_for_k_data, CP_FMZ, 0)

    f_FMZ = np.zeros((128, 128, 18))

    lambda_value = np.log(2) / (20.4 * 60)
    start_index = 0
    for k in range(len(sampling_intervals)):
        end_index = start_index + sampling_intervals[k] - 1
        f_FMZ[:, :, k] = calculate_xm(start_index, end_index, pred_CT_FMZ, lambda_value)
        start_index = end_index + 1

    pred_fmz = f_FMZ / np.max(f_FMZ)
    # pred_fmz = torch.from_numpy(pred_fmz)
    return pred_fmz


def calculate_logan_img(reconstruct_for_k_data, target_forward_k_data, sampling_intervals, cp_data, ki_data_btach,
                        vb_data_batch):
    # 由预测的k1-k4 图像和已知的Cp数据生成预测的18帧的数据
    CP_FMZ = cp_data
    target_forward_k_data = target_forward_k_data.detach().cpu().numpy()
    reconstruct_for_k_data = reconstruct_for_k_data.detach().cpu().numpy()
    ki_data_btach = ki_data_btach.detach().cpu().numpy()
    vb_data_batch = vb_data_batch.detach().cpu().numpy()
    # reconstruct_for_k_data = np.abs(reconstruct_.detach().cpu().numpy()for_k_data)

    pred_CT_FMZ = update_tracer_concentration_np(reconstruct_for_k_data, CP_FMZ, 0)
    target_CT_FMZ = update_tracer_concentration_np(target_forward_k_data, CP_FMZ, 0)

    # assert 0
    integral_Cp = np.cumsum(CP_FMZ, axis=1)  # 对于 PyTorch，cumsum 即为累积积分
    integral_Cp = np.tile(integral_Cp, (128, 128, 1))
    pred_integral_CT = np.cumsum(pred_CT_FMZ, axis=2)
    # target_integral_CT = np.cumsum(target_CT_FMZ, axis=2)
    # x = np.zeros((128, 128, 6))
    # y = np.zeros((128, 128, 6))

    # x_roi = np.where(target_CT_FMZ != 0, integral_Cp / target_CT_FMZ, np.zeros_like(target_CT_FMZ))
    x_roi = np.divide(integral_Cp, target_CT_FMZ, where=target_CT_FMZ != 0, out=np.zeros_like(target_CT_FMZ))

    # x_roi = integral_Cp / target_CT_FMZ
    y_roi = np.divide(pred_integral_CT, pred_CT_FMZ, where=pred_CT_FMZ != 0, out=np.zeros_like(pred_CT_FMZ))
    # y_roi = np.where(pred_CT_FMZ != 0, pred_integral_CT / pred_CT_FMZ, np.zeros_like(pred_CT_FMZ))
    # y_roi = pred_integral_CT / np.clamp(pred_CT_FMZ, min=0.001)
    # y_roi = pred_integral_CT / pred_CT_FMZ

    x = x_roi[:, :, 600:3600]  # Adjust index range for Python (Python uses 0-based indexing)
    y = y_roi[:, :, 600:3600]
    # base = 1800
    # for i in range(0, 6):
    #     index = base + 150
    #     x[:, :, i] = x_roi[:, :, index]
    #     y[:, :, i] = y_roi[:, :, index]
    #     base = base + 300
    ki_data_btach = np.moveaxis(ki_data_btach, 0, -1)  # 在最后一个轴添加维度
    vb_data_batch = np.moveaxis(vb_data_batch, 0, -1)
    # ki_data_btach = ki_data_btach.permute(1, 2, 0)
    # vb_data_batch = vb_data_batch.permute(1, 2, 0)

    target = x * ki_data_btach + vb_data_batch
    pred = y

    pred = pred / np.max(pred)
    target = target / np.max(target)

    # has_nan_or_inf = np.isnan(pred).any() or np.isinf(pred).any()
    # print("reconstruct_for_k_data", has_nan_or_inf)
    # assert 0
    pred = torch.from_numpy(pred).cuda()
    target = torch.from_numpy(target).cuda()
    return pred, target

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
    # k1, k2, k3, k4 = get_mean_k_data_torch(reconstruct_for_k_data)  # 128x128
    k1 = reconstruct_for_k_data[:, :, 0]
    k2 = reconstruct_for_k_data[:, :, 1]
    k3 = reconstruct_for_k_data[:, :, 2]
    k4 = reconstruct_for_k_data[:, :, 3]

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
    discriminant = torch.maximum(discriminant, torch.tensor(0.0))
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
    a = a.unsqueeze(2).repeat(1, 1, T).cuda()
    b = b.unsqueeze(2).repeat(1, 1, T).cuda()
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

    nihe_x = x[:, :, 12:18]
    nihe_y = y[:, :, 12:18]
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