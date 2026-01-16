import torch
from scipy.io import loadmat


def calculate_img_logan3_no_weight(pred_CT_FMZ, target_CT_FMZ, sampling_intervals, cp_data):
    """
        logan_no_weight,loss不加权重
        对后10帧的数据进行计算logan分析损失：y=ax+b方式
        Args:
            pred_CT_FMZ:预测的CT
            target_CT_FMZ:实际的CT
            sampling_intervals:采样协议
            cp_data:血浆数据
            ki_data_btach:logan分析中的斜率
            vb_data_batch:logan分析中的截距

        Returns:预测的和实际的logan分析

        """
    CP_PATH = './data/fmz_zubal_head_sample3_kmin_noise_0307/CP/CP_FMZ.mat'
    cp_data = loadmat(CP_PATH)['CP_FMZ']
    cp_data = cp_data[:, :3600]
    CP_FMZ = torch.from_numpy(cp_data).cuda()
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
        # 创建掩码
        mask_x = CT_values_target != 0
        mask_y = CT_values_pred != 0
        # 使用掩码进行除法
        x_roi = integral_Cp / torch.where(mask_x, CT_values_target, torch.ones_like(CT_values_target))
        y_roi = integral_CT / torch.where(mask_y, CT_values_pred, torch.ones_like(CT_values_pred))

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

            coefficients = torch.linalg.lstsq(X_augmented, y_flat.view(-1, 1)).solution

            # Extract slope (K) and intercept (b)
            K_values[i, j] = coefficients[0].item()
            b_values[i, j] = coefficients[1].item()

    return K_values, b_values