import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import warnings
try:
    import lpips
    # LPIPS模型缓存
    _lpips_cache = {}
except ImportError:
    lpips = None
    warnings.warn("LPIPS package not installed. Install with 'pip install lpips' for LPIPS metric support.")


def cal_mse(target, pred):
    """计算均方误差, 期望输入shape为 (C, H, W)"""
    # 计算平方误差并在所有维度上求平均
    return np.mean(np.square(target - pred), axis=(0, 1)).sum()

def cal_mae(target, pred):
    """计算平均绝对误差, 期望输入shape为 (C, H, W)"""
    # 计算绝对误差并在所有维度上求平均
    return np.mean(np.abs(target - pred), axis=(0, 1)).sum()

def cal_rmse(target, pred):
    """计算均方根误差, 期望输入shape为 (C, H, W)"""
    return np.sqrt(cal_mse(target, pred))

def cal_psnr(target, pred):
    """计算峰值信噪比, 期望输入shape为 (C, H, W)"""
    mse = np.mean(np.square(target - pred))
    if mse == 0:
        return float('inf')
    # 根据图像范围确定最大像素值
    max_pixel = 255.0 if target.max() > (1.0 + 1e-6) else 1.0

    return 10 * np.log10((max_pixel ** 2) / mse)

def cal_ssim(target, pred):
    """计算结构相似性指数, 期望输入shape为 (C, H, W)"""
    # 确定数据范围
    data_range = 255.0 if target.max() > (1.0 + 1e-6) else 1.0
    
    # 转换为 (H, W, C) 格式用于SSIM计算
    if target.ndim == 3:
        target_reshaped = target.transpose(1, 2, 0)
        pred_reshaped = pred.transpose(1, 2, 0)
    else:
        raise ValueError(f"SSIM输入形状应为 (C, H, W), 但得到形状 {target.shape}")
    
    # 处理多通道图像
    if target_reshaped.shape[-1] > 1:
        channel_scores = []
        for c in range(target_reshaped.shape[-1]):
            ssim_score = ssim(target_reshaped[..., c], pred_reshaped[..., c], 
                             data_range=data_range)
            channel_scores.append(ssim_score)
        return np.mean(channel_scores)
    else:
        # 单通道图像
        return ssim(target_reshaped[..., 0], pred_reshaped[..., 0], 
                   data_range=data_range)

def cal_lpips(target, pred, net_type='alex'):
    """计算LPIPS指标, 期望输入shape为 (C, H, W)"""
    if lpips is None:
        warnings.warn("LPIPS package not available. Returning None for LPIPS metric.")
        return None
    
    # 验证输入形状
    if target.ndim != 3:
        raise ValueError(f"LPIPS输入形状应为 (C, H, W), 但得到形状 {target.shape}")
    
    # 转换为 (H, W, C) 格式
    target_reshaped = target.transpose(1, 2, 0)
    pred_reshaped = pred.transpose(1, 2, 0)
    
    # 确保图像是3通道
    if target_reshaped.shape[-1] == 1:
        # 单通道转换为3通道
        target_reshaped = np.repeat(target_reshaped, 3, axis=-1)
        pred_reshaped = np.repeat(pred_reshaped, 3, axis=-1)
    elif target_reshaped.shape[-1] == 2:
        # 2通道转换为3通道, 复制最后一个通道
        target_reshaped = np.concatenate([target_reshaped, target_reshaped[..., -1:]], axis=-1)
        pred_reshaped = np.concatenate([pred_reshaped, pred_reshaped[..., -1:]], axis=-1)
    elif target_reshaped.shape[-1] != 3:
        # 如果不是1,2或3通道, 取前3个通道
        target_reshaped = target_reshaped[..., :3]
        pred_reshaped = pred_reshaped[..., :3]
    
    # 转换为PyTorch tensor并调整格式为NCHW(N=B*T)
    target_tensor = torch.from_numpy(target_reshaped).permute(2, 0, 1).unsqueeze(0).float()
    pred_tensor = torch.from_numpy(pred_reshaped).permute(2, 0, 1).unsqueeze(0).float()
    
    # 归一化到[-1, 1]范围（LPIPS默认期望的范围）
    if target.max() > (1.0 + 1e-6):
        target_tensor = target_tensor / 127.5 - 1.0
        pred_tensor = pred_tensor / 127.5 - 1.0
    else:
        target_tensor = target_tensor * 2.0 - 1.0
        pred_tensor = pred_tensor * 2.0 - 1.0
    
    # 使用缓存的LPIPS模型, 避免每次计算都重新加载模型
    if net_type not in _lpips_cache:
        _lpips_cache[net_type] = lpips.LPIPS(net=net_type, verbose=False)
    loss_fn = _lpips_cache[net_type]
    
    # 计算LPIPS
    with torch.no_grad():
        lpips_score = loss_fn(target_tensor, pred_tensor)
    
    return lpips_score.item()

def cal_metrics(target, pred):
    """计算所有图像质量指标, 形状可以是 (H,W), (C,H,W), (T,C,H,W), (B,T,C,H,W)"""
    # 保存原始维度
    original_shape = target.shape
    original_ndim = target.ndim
    
    # 将输入重塑为 (N, C, H, W) 格式, 其中N是B * T
    # (H, W)
    if original_ndim == 2:
        target_reshaped = target[np.newaxis, np.newaxis, ...]
        pred_reshaped = pred[np.newaxis, np.newaxis, ...]
    # (C, H, W) or (T, H, W)
    elif original_ndim == 3:
        # 假设是通道在前 (C, H, W)
        if original_shape[0] <= 4:
            target_reshaped = target[np.newaxis, ...]
            pred_reshaped = pred[np.newaxis, ...]
        # 假设是时间在前 (T, H, W)
        else:
            target_reshaped = target[..., np.newaxis].transpose(0, 3, 1, 2)
            pred_reshaped = pred[..., np.newaxis].transpose(0, 3, 1, 2)
    # (T, C, H, W)
    elif original_ndim == 4:
        target_reshaped = target.reshape(-1, original_shape[1], original_shape[2], original_shape[3])
        pred_reshaped = pred.reshape(-1, original_shape[1], original_shape[2], original_shape[3])
    # (B, T, C, H, W)
    elif original_ndim == 5:
        target_reshaped = target.reshape(-1, original_shape[2], original_shape[3], original_shape[4])
        pred_reshaped = pred.reshape(-1, original_shape[2], original_shape[3], original_shape[4])
    else:
        raise ValueError(f"Unsupported target dimension: {original_ndim}")
    
    # 计算每个图像的指标
    psnr_scores, ssim_scores, lpips_scores = [], [], []
    
    for i in range(target_reshaped.shape[0]):
        target_img = target_reshaped[i]
        pred_img = pred_reshaped[i]
        
        psnr_scores.append(cal_psnr(target_img, pred_img))
        ssim_scores.append(cal_ssim(target_img, pred_img))
        
        if lpips is not None:
            lpips_scores.append(cal_lpips(target_img, pred_img))
    
    # 计算平均指标
    metrics = {
        "mse": cal_mse(target, pred),
        "mae": cal_mae(target, pred),
        "rmse": cal_rmse(target, pred),
        "psnr": np.mean(psnr_scores),
        "ssim": np.mean(ssim_scores)
    }
    
    # 添加LPIPS指标(如果可用）
    if lpips is not None and len(lpips_scores) > 0:
        metrics["lpips"] = np.mean(lpips_scores)
    
    return metrics