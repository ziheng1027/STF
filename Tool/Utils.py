import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def get_trainer(model_name, dataset_name, config, datasets_config):
    try:
        module_name = f"Trainer.Trainer_{model_name}"
        module = __import__(module_name, fromlist=["Trainer"])
        return module.Trainer(config, datasets_config, dataset_name)
    except ImportError:
        raise ValueError(f"未找到Trainer: {module_name}")

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
    plt.figure(figsize=(15, 6))
    # input
    for t in range(T):
        plt.subplot(3, T, t+1)
        plt.imshow(input[t, 0, :, :])
        plt.tick_params(axis='both', which='both', length=0)
        if t == 0:
            plt.ylabel("input", rotation=0, labelpad=20, ha='right', va='center')
    # target
    for t in range(T):
        plt.subplot(3, T, t+1+T)
        plt.imshow(target[t, 0, :, :])
        plt.tick_params(axis='both', which='both', length=0)
        if t == 0:
            plt.ylabel("target", rotation=0, labelpad=20, ha='right', va='center')
    # output
    for t in range(T):
        plt.subplot(3, T, t+1+2*T)
        plt.imshow(output[t, 0, :, :])
        plt.tick_params(axis='both', which='both', length=0)
        if t == 0:
            plt.ylabel("output", rotation=0, labelpad=20, ha='right', va='center')
    
    plt.suptitle(f"-Sample {idx}-", y=0.98, fontsize=16)
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
