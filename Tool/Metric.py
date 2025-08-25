import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import warnings
try:
    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
    lpips_available = True
except ImportError:
    lpips_available = False
    warnings.warn("torchmetrics package not installed. Install with 'pip install torchmetrics' for LPIPS metric support.")

try:
    import lpips
    original_lpips_available = True
except ImportError:
    original_lpips_available = False


def cal_mse(target, pred):
    """计算MSE，支持2D-5D输入，返回单张图像的总误差"""
    if target.ndim == 5:
        # (B, T, C, H, W) -> 计算每个样本的总误差
        return np.sum(np.square(target - pred), axis=(1, 2, 3, 4))
    elif target.ndim == 4:
        # (T, C, H, W) -> 计算序列的总误差
        return np.sum(np.square(target - pred), axis=(0, 1, 2, 3))
    else:
        # 2D或3D图像
        return np.sum(np.square(target - pred))

def cal_mae(target, pred):
    """计算MAE，支持2D-5D输入，返回单张图像的总绝对误差"""
    if target.ndim == 5:
        # (B, T, C, H, W) -> 计算每个样本的总绝对误差
        return np.sum(np.abs(target - pred), axis=(1, 2, 3, 4))
    elif target.ndim == 4:
        # (T, C, H, W) -> 计算序列的总绝对误差
        return np.sum(np.abs(target - pred), axis=(0, 1, 2, 3))
    else:
        # 2D或3D图像
        return np.sum(np.abs(target - pred))

def cal_rmse(target, pred):
    """计算RMSE"""
    return np.sqrt(cal_mse(target, pred))

def cal_psnr(target, pred):
    """计算PSNR，支持2D-5D输入"""
    mse = cal_mse(target, pred)
    if np.any(mse == 0):
        # 处理完美重建的情况
        return np.where(mse == 0, float('inf'), 0)
    
    # 根据图像范围确定最大像素值
    max_pixel = 255.0 if target.max() > 1.0 else 1.0
    
    # 计算PSNR
    psnr_values = 10 * np.log10((max_pixel ** 2) / mse)
    
    # 如果是批量数据，返回平均值
    if target.ndim == 5:
        return np.mean(psnr_values)
    else:
        return psnr_values

def cal_ssim(target, pred):
    """计算SSIM，支持2D-5D输入"""
    # 确定数据范围
    data_range = 255.0 if target.max() > 1.0 else 1.0
    
    if target.ndim == 5:
        # (B, T, C, H, W) -> 对每个样本的每个时间帧计算SSIM
        ssim_scores = []
        for b in range(target.shape[0]):
            for t in range(target.shape[1]):
                frame_target = target[b, t]
                frame_pred = pred[b, t]
                if frame_target.ndim == 3 and frame_target.shape[2] > 1:
                    # 多通道图像
                    channel_scores = []
                    for c in range(frame_target.shape[2]):
                        ssim_score = ssim(frame_target[..., c], frame_pred[..., c], data_range=data_range)
                        channel_scores.append(ssim_score)
                    ssim_scores.append(np.mean(channel_scores))
                else:
                    # 单通道图像
                    ssim_score = ssim(frame_target, frame_pred, data_range=data_range)
                    ssim_scores.append(ssim_score)
        return np.mean(ssim_scores)
    elif target.ndim == 4:
        # (T, C, H, W) -> 对每个时间帧计算SSIM
        ssim_scores = []
        for t in range(target.shape[0]):
            frame_target = target[t]
            frame_pred = pred[t]
            if frame_target.ndim == 3 and frame_target.shape[2] > 1:
                channel_scores = []
                for c in range(frame_target.shape[2]):
                    ssim_score = ssim(frame_target[..., c], frame_pred[..., c], data_range=data_range)
                    channel_scores.append(ssim_score)
                ssim_scores.append(np.mean(channel_scores))
            else:
                ssim_score = ssim(frame_target, frame_pred, data_range=data_range)
                ssim_scores.append(ssim_score)
        return np.mean(ssim_scores)
    else:
        # 2D或3D图像
        if target.ndim == 3 and target.shape[2] > 1:
            # 多通道图像
            ssim_scores = []
            for c in range(target.shape[2]):
                ssim_score = ssim(target[..., c], pred[..., c], data_range=data_range)
                ssim_scores.append(ssim_score)
            return np.mean(ssim_scores)
        else:
            # 单通道图像
            return ssim(target, pred, data_range=data_range)

def cal_lpips(target, pred, net_type='alex'):
    """计算LPIPS指标，支持2D-5D输入"""
    if target.ndim == 5:
        # (B, T, C, H, W) -> 对每个样本的每个时间帧计算LPIPS并取平均
        lpips_scores = []
        for b in range(target.shape[0]):
            for t in range(target.shape[1]):
                frame_target = target[b, t]
                frame_pred = pred[b, t]
                score = _cal_single_lpips(frame_target, frame_pred, net_type)
                if score is not None:
                    lpips_scores.append(score)
        return np.mean(lpips_scores) if lpips_scores else None
    elif target.ndim == 4:
        # (T, C, H, W) -> 对每个时间帧计算LPIPS并取平均
        lpips_scores = []
        for t in range(target.shape[0]):
            frame_target = target[t]
            frame_pred = pred[t]
            score = _cal_single_lpips(frame_target, frame_pred, net_type)
            if score is not None:
                lpips_scores.append(score)
        return np.mean(lpips_scores) if lpips_scores else None
    else:
        # 2D或3D图像
        return _cal_single_lpips(target, pred, net_type)

def _cal_single_lpips(target, pred, net_type='alex'):
    """计算单帧图像的LPIPS指标"""
    # 优先使用torchmetrics的LPIPS实现
    if lpips_available:
        return _cal_lpips_torchmetrics(target, pred, net_type)
    elif original_lpips_available:
        return _cal_lpips_original(target, pred, net_type)
    else:
        warnings.warn("LPIPS package not available. Returning None for LPIPS metric.")
        return None

def _cal_lpips_torchmetrics(target, pred, net_type='alex'):
    """使用torchmetrics的LPIPS实现"""
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
    
    # 归一化到[0, 1]范围（torchmetrics LPIPS期望的范围）
    if target.max() > 1.0:
        target_tensor = target_tensor / 255.0
        pred_tensor = pred_tensor / 255.0
    
    # 初始化torchmetrics LPIPS模型
    # torchmetrics使用不同的网络名称映射
    net_mapping = {'alex': 'alex', 'vgg': 'vgg', 'squeeze': 'squeezenet'}
    tm_net_type = net_mapping.get(net_type, 'alex')
    
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=tm_net_type)
    
    # 计算LPIPS
    with torch.no_grad():
        lpips_score = lpips_metric(target_tensor, pred_tensor)
    
    return lpips_score.item()

def _cal_lpips_original(target, pred, net_type='alex'):
    """使用原始lpips包的实现"""
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
    
    # 归一化到[-1, 1]范围（原始LPIPS默认期望的范围）
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
    if lpips_available or original_lpips_available:
        metrics["lpips"] = cal_lpips(target, pred)
    
    return metrics