# Tool/Utils.py
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def get_trainer(model_name, dataset_name, model_config, dataset_config, metric_config):
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
            np.save(f"{sample_dir}/{model_name}-sample_{idx}.npy", samples)
        idx += 1

    return idx

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