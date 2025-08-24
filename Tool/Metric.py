import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import warnings
try:
    import lpips
except ImportError:
    lpips = None
    warnings.warn("LPIPS package not installed. Install with 'pip install lpips' for LPIPS metric support.")


def cal_mse(target, pred):
    return np.mean(np.square(target - pred), axis=(0, 1)).sum()

def cal_mae(target, pred):
    return np.mean(np.abs(target - pred), axis=(0, 1)).sum()

def cal_rmse(target, pred):
    return np.sqrt(cal_mse(target, pred))

def cal_psnr(target, pred):
    mse = cal_mse(target, pred)
    if mse == 0:
        return float('inf')
    # 根据图像范围确定最大像素值
    max_pixel = 255.0 if target.max() > 1.0 else 1.0

    return 10 * np.log10((max_pixel ** 2) / mse)

def cal_ssim(target, pred):
    # 确定数据范围
    data_range = 255.0 if target.max() > 1.0 else 1.0
    # 多通道图像, 计算每个通道的SSIM并取平均
    if target.ndim == 3 and target.shape[2] > 1:
        ssim_scores = []
        for c in range(target.shape[2]):
            ssim_score = ssim(target[..., c], pred[..., c], data_range=data_range)
            ssim_scores.append(ssim_score)
        return np.mean(ssim_scores)
    else:
        # 单通道图像
        return ssim(target, pred, data_range=data_range)

def cal_lpips(target, pred, net_type='alex'):
    """计算LPIPS指标, 需要输入为numpy数组, 形状为(H, W, C)或(H, W)"""
    if lpips is None:
        warnings.warn("LPIPS package not available. Returning None for LPIPS metric.")
        return None
    
    # 确保输入是3通道图像
    if target.ndim == 2:
        # 灰度图像转换为3通道
        target = np.stack([target] * 3, axis=-1)
        pred = np.stack([pred] * 3, axis=-1)
    elif target.ndim == 3 and target.shape[2] == 1:
        # 单通道转换为3通道
        target = np.repeat(target, 3, axis=2)
        pred = np.repeat(pred, 3, axis=2)
    
    # 转换为PyTorch tensor并调整格式为NCHW
    target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float()
    pred_tensor = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float()
    
    # 归一化到[-1, 1]范围（LPIPS默认期望的范围）
    if target.max() > 1.0:
        target_tensor = target_tensor / 127.5 - 1.0
        pred_tensor = pred_tensor / 127.5 - 1.0
    else:
        target_tensor = target_tensor * 2.0 - 1.0
        pred_tensor = pred_tensor * 2.0 - 1.0
    
    # 初始化LPIPS模型
    loss_fn = lpips.LPIPS(net=net_type)
    
    # 计算LPIPS
    with torch.no_grad():
        lpips_score = loss_fn(target_tensor, pred_tensor)
    
    return lpips_score.item()

def cal_metrics(target, pred):
    metrics = {
        "mse": cal_mse(target, pred),
        "mae": cal_mae(target, pred),
        "rmse": cal_rmse(target, pred),
        "psnr": cal_psnr(target, pred),
        "ssim": cal_ssim(target, pred)
    }
    
    # 添加LPIPS指标（如果可用）
    if lpips is not None:
        metrics["lpips"] = cal_lpips(target, pred)
    
    return metrics