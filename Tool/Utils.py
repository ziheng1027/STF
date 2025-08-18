import os
import torch
import numpy as np


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
            np.save(f"{sample_dir}/{model_name}_sample{idx}_batch{i}.npy", samples)
        idx += 1

    return idx

def visualize_sample_figure():
    """可视化样本图像"""
    pass

def visualize_sample_gif():
    """可视化样本GIF"""
    pass