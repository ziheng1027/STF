import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def get_trainer(model_name, dataset_name, config, datasets_config):
    try:
        module_name = f"Trainer.Trainer_{model_name}"
        module = __import__(module_name, fromlist=["Trainer"])
        return module.Trainer(config, datasets_config, dataset_name)
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
    # 使用base_seed + worker_id来确保每个worker有不同但确定的随机性
    base_seed = 42  # 这个值应该与main.py中设置的seed一致
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)

def save_test_samples(idx, input, target, output, model_name, sample_dir, interval):
    """保存模型输出样本"""
    batch_size = input.shape[0]
    for i in range(batch_size):
        if idx % interval == 0:
            samples = {
                "input": input[i].cpu().numpy(),
                "target": target[i].cpu().numpy(),
                "output": output[i].cpu().numpy()
            }
            np.save(f"{sample_dir}/{model_name}_sample_{idx}.npy", samples)
        idx += 1

    return idx

def visualize_base(idx, input, target, output):
    """可视化"""
    T, C, _, _ = input.shape
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
    induices = set()
    # 提取样本序号
    for file in os.listdir(dir):
        if file.endswith(".npy"):
            idx = int(file.split("_")[2].split(".")[0])
            induices.add(idx)
    induices = sorted(list(induices))

    for idx in induices:
        sample = np.load(f"{dir}/{model_name}_sample_{idx}.npy", allow_pickle=True).item()
        input = sample["input"]
        target = sample["target"]
        output = sample["output"]
        visualize_base(idx, input, target, output)

def visualize_gif():
    """可视化为GIF"""
    pass


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