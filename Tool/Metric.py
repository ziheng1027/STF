import torch
import numpy as np
import warnings
from skimage.metrics import structural_similarity as ssim
from Tool.Utils import reshape_to_nchw
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
        target_hwc = target.transpose(1, 2, 0)
        pred_hwc = pred.transpose(1, 2, 0)
    else:
        raise ValueError(f"SSIM输入形状应为 (C, H, W), 但得到形状 {target.shape}")
    
    # 处理多通道图像
    if target_hwc.shape[-1] > 1:
        channel_scores = []
        for c in range(target_hwc.shape[-1]):
            ssim_score = ssim(target_hwc[..., c], pred_hwc[..., c], 
                             data_range=data_range)
            channel_scores.append(ssim_score)
        return np.mean(channel_scores)
    else:
        # 单通道图像
        return ssim(target_hwc[..., 0], pred_hwc[..., 0], 
                   data_range=data_range)

def cal_lpips(target, pred, net_type='alex'):
    """计算LPIPS指标, 期望输入shape为 (C, H, W)"""
    if lpips is None:
        warnings.warn("LPIPS package 不可用. LPIPS指标为None.")
        return None
    
    # 验证输入形状
    if target.ndim != 3:
        raise ValueError(f"LPIPS输入形状应为 (C, H, W), 但得到形状 {target.shape}")
    
    # 转换为 (H, W, C)
    target_hwc = target.transpose(1, 2, 0)
    pred_hwc = pred.transpose(1, 2, 0)
    
    # 确保图像是3通道
    if target_hwc.shape[-1] == 1:
        # 单通道转换为3通道
        target_hwc = np.repeat(target_hwc, 3, axis=-1)
        pred_hwc = np.repeat(pred_hwc, 3, axis=-1)
    elif target_hwc.shape[-1] == 2:
        # 2通道转换为3通道, 复制最后一个通道
        target_hwc = np.concatenate([target_hwc, target_hwc[..., -1:]], axis=-1)
        pred_hwc = np.concatenate([pred_hwc, pred_hwc[..., -1:]], axis=-1)
    elif target_hwc.shape[-1] != 3:
        # 如果不是1,2或3通道, 取前3个通道
        target_hwc = target_hwc[..., :3]
        pred_hwc = pred_hwc[..., :3]
    
    # 转换为PyTorch tensor并调整格式为NCHW(N=B*T)
    target_tensor = torch.from_numpy(target_hwc).permute(2, 0, 1).unsqueeze(0).float()
    pred_tensor = torch.from_numpy(pred_hwc).permute(2, 0, 1).unsqueeze(0).float()
    
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

def cal_binary_mask(target, pred, threshold):
    """计算二值掩码"""
    target_mask = (target >= threshold).astype(int)
    pred_mask = (pred >= threshold).astype(int)
    return target_mask, pred_mask

def cal_precision(target_mask, pred_mask):
    """计算精确率"""
    tp = np.sum(target_mask * pred_mask)
    fp = np.sum((1 - target_mask) * pred_mask)
    return tp / (tp + fp + 1e-6)

def cal_recall(target_mask, pred_mask):
    """计算召回率"""
    tp = np.sum(target_mask * pred_mask)
    fn = np.sum(target_mask * (1 - pred_mask))
    return tp / (tp + fn + 1e-6)

def cal_f1_score(target_mask, pred_mask):
    """计算F1分数"""
    precision = cal_precision(target_mask, pred_mask)
    recall = cal_recall(target_mask, pred_mask)
    return (2 * precision * recall) / (precision + recall + 1e-6)

def cal_accuracy(target_mask, pred_mask):
    """计算准确率"""
    tp = np.sum(target_mask * pred_mask)
    tn = np.sum((1 - target_mask) * (1 - pred_mask))
    fp = np.sum((1 - target_mask) * pred_mask)
    fn = np.sum(target_mask * (1 - pred_mask))
    return (tp + tn) / (tp + tn + fp + fn + 1e-6)

def cal_far(target_mask, pred_mask):
    """计算空报率"""
    fp = np.sum((1 - target_mask) * pred_mask)
    tn = np.sum((1 - target_mask) * (1 - pred_mask))
    return fp / (fp + tn + 1e-6)

def cal_csi(target_mask, pred_mask):
    """计算临界成功指数"""
    tp = np.sum(target_mask * pred_mask)
    fp = np.sum((1 - target_mask) * pred_mask)
    fn = np.sum(target_mask * (1 - pred_mask))
    return tp / (tp + fp + fn + 1e-6)

def cal_hss(target_mask, pred_mask):
    """计算Heidke skill score"""
    tp = np.sum(target_mask * pred_mask)
    tn = np.sum((1 - target_mask) * (1 - pred_mask))
    fp = np.sum((1 - target_mask) * pred_mask)
    fn = np.sum(target_mask * (1 - pred_mask))
    num = (tp * tn) - (fp * fn)
    den = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    return 2 * num / (den + 1e-6)

def cal_metrics(target, pred, metrics_name=None, threshold=None):
    """计算所有图像质量指标, 形状可以是 (H,W), (C,H,W), (T,C,H,W), (B,T,C,H,W)"""
    # 如果没有指定需要计算哪些指标, 则计算默认指标
    if metrics_name is None:
        metrics_name = ["MSE", "MAE", "RMSE", "PSNR", "SSIM", "LPIPS"]
    
    # 将输入转换为(N, C, H, W)的形式, N=B*T
    target_nchw, pred_nchw = reshape_to_nchw(target, pred)
    
    # 计算每帧图像的指标
    psnr_scores, ssim_scores, lpips_scores = [], [], []
    metrics = {}
    
    for i in range(target_nchw.shape[0]):
        target_img = target_nchw[i]
        pred_img = pred_nchw[i]
        
        if "PSNR" in metrics_name:
            psnr_scores.append(cal_psnr(target_img, pred_img))
        if "SSIM" in metrics_name:
            ssim_scores.append(cal_ssim(target_img, pred_img))
        if "LPIPS" in metrics_name and lpips is not None:
            lpips_scores.append(cal_lpips(target_img, pred_img))
    
    if "MSE" in metrics_name:
        metrics["MSE"] = cal_mse(target, pred)
    if "MAE" in metrics_name:
        metrics["MAE"] = cal_mae(target, pred)
    if "RMSE" in metrics_name:
        metrics["RMSE"] = cal_rmse(target, pred)
    if "PSNR" in metrics_name:
        metrics["PSNR"] = np.mean(psnr_scores)
    if "SSIM" in metrics_name:
        metrics["SSIM"] = np.mean(ssim_scores)
    if "LPIPS" in metrics_name and lpips is not None and lpips_scores:
        metrics["LPIPS"] = np.mean(lpips_scores)
    
    # 计算二值化指标(气象数据集)
    if threshold is not None:
        target_mask, pred_mask = cal_binary_mask(target, pred, threshold)
        if "Precision" in metrics_name:
            metrics["Precision"] = cal_precision(target_mask, pred_mask)
        if "Recall" in metrics_name:
            metrics["Recall"] = cal_recall(target_mask, pred_mask)
        if "F1_Score" in metrics_name:
            metrics["F1_Score"] = cal_f1_score(target_mask, pred_mask)
        if "Accuracy" in metrics_name:
            metrics["Accuracy"] = cal_accuracy(target_mask, pred_mask)
        if "FAR" in metrics_name:
            metrics["FAR"] = cal_far(target_mask, pred_mask)
        if "CSI" in metrics_name:
            metrics["CSI"] = cal_csi(target_mask, pred_mask)
        if "HSS" in metrics_name:
            metrics["HSS"] = cal_hss(target_mask, pred_mask)
    
    return metrics