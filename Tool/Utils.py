# Tool/Utils.py
import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


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

def worker_init_fn(worker_id, base_seed=42):
    """DataLoader worker初始化函数, 确保每个worker有独立的随机种子"""
    # 使用base_seed + worker_id来确保每个worker有不同但确定的随机性
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)

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

def get_scheduled_sampling_mask(iters, input_patched, input_frames, start_iters=20000, end_iters=50000, reverse=False):
    """计算Scheduled Sampling策略的概率阈值"""
    B, T, C, H, W = input_patched.shape
    if end_iters <= start_iters: # prob < threshold => GT
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
        # output部分概率从1.0线性衰减到0
        if iters < end_iters:
            output_threshold = max(0.0, 1.0 - (1.0 / end_iters) * iters)
        else:
            output_threshold = 0.0
    
    # 生成掩码
    mask_patched = torch.zeros((B, T - 1, 1, 1, 1), device=input_patched.device)
    random_prob = torch.rand((B, T - 1), device=input_patched.device)

    # 划分input和output部分
    split_idx = input_frames - 1
    mask_patched[:, :split_idx] = (random_prob[:, :split_idx] < input_threshold).float().view(B, split_idx, 1, 1, 1)
    mask_patched[:, split_idx:] = (random_prob[:, split_idx:] < output_threshold).float().view(B, T - 1 - split_idx, 1, 1, 1)
    
    return mask_patched

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

"""可视化相关"""
def plot_loss(train_losses, val_losses, model_name, dataset_name):
    """绘制训练和验证损失曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'r-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'g-', label='Validation Loss', linewidth=2)
    plt.title(f'{model_name} on {dataset_name} - Loss Curves', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.savefig(f"Output/Visualization/{dataset_name}/{model_name}/loss_curve.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"损失曲线已保存至: Output/Visualization/{dataset_name}/{model_name}/loss_curve.pdf")

def visualize_base(idx, input, target, output):
    """可视化"""
    T, C, _, _ = target.shape
    plt.figure(figsize=(15, 9))
    # input
    for t in range(T):
        plt.subplot(4, T, t+1)
        plt.imshow(input[t, 0, :, :])
        plt.tick_params(axis='both', which='both', length=0)
        if t == 0:
            plt.ylabel("input", rotation=0, labelpad=20, ha='right', va='center')
        plt.axis('off')
    # target
    for t in range(T):
        plt.subplot(4, T, t+1+T)
        plt.imshow(target[t, 0, :, :])
        plt.tick_params(axis='both', which='both', length=0)
        if t == 0:
            plt.ylabel("target", rotation=0, labelpad=20, ha='right', va='center')
    # output
    for t in range(T):
        plt.subplot(4, T, t+1+2*T)
        plt.imshow(output[t, 0, :, :])
        plt.tick_params(axis='both', which='both', length=0)
        if t == 0:
            plt.ylabel("output", rotation=0, labelpad=20, ha='right', va='center')
    # error
    for t in range(T):
        plt.subplot(4, T, t+1+3*T)
        plt.imshow(output[t, 0, :, :] - target[t, 0, :, :])
        plt.tick_params(axis='both', which='both', length=0)
        if t == 0:
            plt.ylabel("error", rotation=0, labelpad=20, ha='right', va='center')
    
    plt.suptitle(f"|-Sample {idx}-|", y=0.98, fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_figure(model_name, dataset_name):
    """可视化为图像"""
    dir = f"Output/Sample/{dataset_name}/{model_name}"
    indices = set()
    # 提取样本序号
    for file in os.listdir(dir):
        if file.endswith(".npy"):
            idx = int(file.split("_")[1].split(".")[0])
            indices.add(idx)
    indices = sorted(list(indices))

    for idx in indices:
        sample = np.load(f"{dir}/{model_name}-sample_{idx}.npy", allow_pickle=True).item()
        input = sample["input"]
        target = sample["target"]
        output = sample["output"]
        visualize_base(idx, input, target, output)

def visualize_gif():
    """可视化为GIF"""
    pass
