import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.cluster import normalized_mutual_info_score


def calculate_cs(image1, image2):
    # 将图像展平为一维数组
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()

    # 计算余弦相似度
    similarity = 1 - cosine(image1_flat, image2_flat)
    return similarity


def calculate_nmi(matrix1, matrix2):
    # 将矩阵转换为整数类型的标签
    labels1 = np.round(matrix1).astype(int).flatten()
    labels2 = np.round(matrix2).astype(int).flatten()

    # 计算规范化互信息
    nmi = normalized_mutual_info_score(labels1, labels2)
    return nmi
