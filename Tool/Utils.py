# Tool/Utils.py
import math
import torch
import random
import numpy as np
import torch.nn as nn


"""训练相关"""
def get_trainer(model_name, dataset_name, model_config, dataset_config, metric_config):
    """获取指定模型的训练器"""
    try:
        module_name = f"Trainer.Trainer_{model_name}"
        module = __import__(module_name, fromlist=["Trainer"])
        return module.Trainer(model_config, dataset_config, metric_config, dataset_name)
    except ImportError:
        raise ValueError(f"未找到Trainer: {module_name}")

def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    """DataLoader worker初始化函数, 确保每个worker有独立的随机种子"""
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def patchify(img, patch_size):
        """将输入图像切分为patch, 然后将每个patch合并到通道维度"""
        B, T, C, H, W = img.shape
        # [B, T, C, H, W] -> [B, T, C, H/patch_size, patch_size, W/patch_size, patch_size]
        img_patched = img.reshape(
            B, T, C, H // patch_size, patch_size, W // patch_size, patch_size
        ).permute(0, 1, 2, 4, 6, 3, 5).reshape(
            B, T, C * patch_size * patch_size, H // patch_size, W // patch_size
        )
        return img_patched
    
def unpatchify(img_patched, patch_size):
    """将patch合并回原始图像尺寸"""
    B, T, C_patched, H_patched, W_patched = img_patched.shape
    C = C_patched // (patch_size * patch_size)
    img = img_patched.reshape(
        B, T, C, patch_size, patch_size, H_patched, W_patched
    ).permute(0, 1, 2, 5, 3, 6, 4).reshape(
        B, T, C, H_patched * patch_size, W_patched * patch_size
    )
    return img

def get_scheduled_sampling_mask(iters, input_patched, input_frames, start_iters=25000, end_iters=50000, reverse=False, eta=1.0, sampling_changing_rate=0.00002):
    """计算Scheduled Sampling策略的概率阈值"""
    B, T, C, H, W = input_patched.shape
    if end_iters <= start_iters:
        raise ValueError("end_iters必须大于start_iters")

    if reverse:
        # input部分概率从0.5指数增长到1.0
        if iters < start_iters:
            input_threshold = 0.5
        elif iters < end_iters:
            input_threshold = 1.0 - 0.5 * math.exp(-float(iters - start_iters) / (end_iters - start_iters))
        else:
            input_threshold = 1.0

        # output部分概率从0.5线性衰减到0
        if iters < start_iters:
            output_threshold = 0.5
        elif iters < end_iters:
            output_threshold = 0.5 - (0.5 / (end_iters - start_iters)) * (iters - start_iters)
        else:
            output_threshold = 0.0
    else:
        # input部分概率始终为1.0
        input_threshold = 1.0

        # output部分概率固定步长递减
        if iters < end_iters:
            eta -= sampling_changing_rate
            eta = max(0.0, eta)  # 确保不低于0
        else:
            eta = 0.0
        output_threshold = eta

    # 生成掩码
    mask_patched = torch.zeros((B, T - 1, 1, 1, 1), device=input_patched.device)
    random_prob = torch.rand((B, T - 1), device=input_patched.device)

    # 划分input和output部分
    split_idx = input_frames - 1
    mask_patched[:, :split_idx] = (random_prob[:, :split_idx] < input_threshold).float().view(B, split_idx, 1, 1, 1)
    mask_patched[:, split_idx:] = (random_prob[:, split_idx:] < output_threshold).float().view(B, T - 1 - split_idx, 1, 1, 1)

    return eta, mask_patched

def drop_path(x, drop_prob=0, training=False, scale_by_keep=True):
    """Stochastic Depth, 在训练深度网络时随机丢弃残差块的主路径，从而减少过拟合并提升泛化能力"""
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    # 生成与输入张量维度匹配的随机张量
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # 使用bernoulli_分布生成0/1掩码
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

    # 如果开启scale_by_keep, 则对剩余路径进行缩放, 以保持期望值一致
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    
    return x * random_tensor


class DropPath(nn.Module):
    """DropPath层, 用于在模型中作为一个标准的层插件使用"""
    def __init__(self, drop_prob=0, scale_by_keep=True):
        super().__init__()

        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3): 0.3f}"


"""测试相关"""
def save_test_samples(idx, input, target, output, model_name, sample_dir, interval):
    """保存测试样本以便可视化"""
    batch_size = input.shape[0]
    for i in range(batch_size):
        if idx % interval == 0:
            samples = {
                "input": input[i].cpu().numpy(),
                "target": target[i].cpu().numpy(),
                "output": output[i].cpu().numpy()
            }
            np.save(f"{sample_dir}/{model_name}-sample_{idx}.npy", samples)
        idx += 1

    return idx

def reshape_to_nchw(target, pred):
    """将输入重塑为 (N, C, H, W) 格式, 其中N是B * T"""
    # 原始维度
    shape = target.shape
    ndim = target.ndim
    # (H, W), 添加N和C维度
    if ndim == 2:
        target_reshaped = target[np.newaxis, np.newaxis, ...]
        pred_reshaped = pred[np.newaxis, np.newaxis, ...]
    # (C, H, W) or (T, H, W)
    elif ndim == 3:
        # 假设是通道在前 (C, H, W), 添加N维度
        if shape[0] < 4:
            target_reshaped = target[np.newaxis, ...]
            pred_reshaped = pred[np.newaxis, ...]
        # 假设是时间在前 (T, H, W), 添加C维度
        else:
            target_reshaped = target[..., np.newaxis].transpose(0, 3, 1, 2)
            pred_reshaped = pred[..., np.newaxis].transpose(0, 3, 1, 2)
    # (T, C, H, W)
    elif ndim == 4:
        target_reshaped = target.reshape(-1, shape[1], shape[2], shape[3])
        pred_reshaped = pred.reshape(-1, shape[1], shape[2], shape[3])
    # (B, T, C, H, W)
    elif ndim == 5:
        target_reshaped = target.reshape(-1, shape[2], shape[3], shape[4])
        pred_reshaped = pred.reshape(-1, shape[2], shape[3], shape[4])
    else:
        raise ValueError(f"不支持的输入维度: {ndim}")
    
    return target_reshaped, pred_reshaped
