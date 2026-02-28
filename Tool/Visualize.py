import os
import numpy as np
import matplotlib.pyplot as plt
from Tool.ColorMap import get_cmap


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

def visualize_movingmnist(idx, input, target, output):
    """可视化MovingMNIST数据集的预测结果"""
    T, C, _, _ = target.shape
    cmap, _, vmin, vmax = get_cmap('MovingMNIST')

    plt.figure(figsize=(18, 9))

    # input行
    for t in range(T):
        ax = plt.subplot(4, T, t + 1)
        im = ax.imshow(input[t, 0], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={t + 1}", fontsize=10)
        if t == T - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if t == 0:
            ax.set_ylabel("Input", rotation=90, labelpad=15, ha='center', va='center', fontsize=18)

    # target行
    for t in range(T):
        ax = plt.subplot(4, T, t + 1 + T)
        im = ax.imshow(target[t, 0], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={T + t + 1}", fontsize=10)
        if t == T - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if t == 0:
            ax.set_ylabel("Target", rotation=90, labelpad=15, ha='center', va='center', fontsize=18)

    # output行
    for t in range(T):
        ax = plt.subplot(4, T, t + 1 + 2 * T)
        im = ax.imshow(output[t, 0], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if t == T - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if t == 0:
            ax.set_ylabel("Output", rotation=90, labelpad=15, ha='center', va='center', fontsize=18)

    # error行
    for t in range(T):
        ax = plt.subplot(4, T, t + 1 + 3 * T)
        error = output[t, 0] - target[t, 0]
        v_max = max(abs(error.min()), abs(error.max()))
        im = ax.imshow(error, cmap=cmap, vmin=-v_max, vmax=v_max)
        ax.set_xticks([])
        ax.set_yticks([])
        if t == T - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if t == 0:
            ax.set_ylabel("Error", rotation=90, labelpad=15, ha='center', va='center', fontsize=18)

    plt.suptitle(f"Sample {idx}", y=0.98, fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05, wspace=0.2, hspace=0.2)
    plt.show()


def visualize_taxibj(idx, input, target, output):
    """可视化TaxiBJ数据集的预测结果"""
    T, C, _, _ = target.shape
    cmap, _, vmin, vmax = get_cmap('TaxiBJ')

    plt.figure(figsize=(12, 9))

    # input行
    for t in range(T):
        ax = plt.subplot(4, T, t + 1)
        im = ax.imshow(input[t, 0], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={t + 1}", fontsize=10)
        if t == T - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if t == 0:
            ax.set_ylabel("Input", rotation=90, labelpad=15, ha='center', va='center', fontsize=18)

    # target行
    for t in range(T):
        ax = plt.subplot(4, T, t + 1 + T)
        im = ax.imshow(target[t, 0], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={T + t + 1}", fontsize=10)
        if t == T - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if t == 0:
            ax.set_ylabel("Target", rotation=90, labelpad=15, ha='center', va='center', fontsize=18)

    # output行
    for t in range(T):
        ax = plt.subplot(4, T, t + 1 + 2 * T)
        im = ax.imshow(output[t, 0], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if t == T - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if t == 0:
            ax.set_ylabel("Output", rotation=90, labelpad=15, ha='center', va='center', fontsize=18)

    # error行
    for t in range(T):
        ax = plt.subplot(4, T, t + 1 + 3 * T)
        error = output[t, 0] - target[t, 0]
        v_max = max(abs(error.min()), abs(error.max()))
        im = ax.imshow(error, cmap=cmap, vmin=-v_max, vmax=v_max)
        ax.set_xticks([])
        ax.set_yticks([])
        if t == T - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if t == 0:
            ax.set_ylabel("Error", rotation=90, labelpad=15, ha='center', va='center', fontsize=18)

    plt.suptitle(f"Sample {idx}", y=0.98, fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05, wspace=0.2, hspace=0.2)
    plt.show()


def visualize_sevir(idx, input, target, output):
    """可视化SEVIR数据集的预测结果, 使用VIL专用颜色映射"""
    T_in, C, H, W = input.shape
    T_out = target.shape[0]

    # 获取VIL颜色映射
    vil_cmap, vil_norm, vil_vmin, vil_vmax = get_cmap('SEVIR', 'vil')

    # SEVIR数据集被归一化到[0,1], 需要乘回255以匹配VIL colormap的边界
    input = input * 255.0
    target = target * 255.0
    output = output * 255.0

    fig, axes = plt.subplots(3, T_out, figsize=(3 * T_out, 9))

    # 绘制input帧(显示最后T_out帧用于对比)
    for t in range(T_out):
        ax = axes[0, t] if T_out > 1 else axes[0]
        input_idx = T_in - T_out + t
        im = ax.imshow(input[input_idx, 0], cmap=vil_cmap, norm=vil_norm,
                      vmin=vil_vmin, vmax=vil_vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={input_idx + 1}", fontsize=10)
        if t == T_out - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if t == 0:
            ax.set_ylabel("Input", rotation=90, labelpad=15, ha='center', va='center', fontsize=18)

    # 绘制target帧
    for t in range(T_out):
        ax = axes[1, t] if T_out > 1 else axes[1]
        im = ax.imshow(target[t, 0], cmap=vil_cmap, norm=vil_norm, vmin=vil_vmin, vmax=vil_vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={T_in + t + 1}", fontsize=10)
        if t == T_out - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if t == 0:
            ax.set_ylabel("Target", rotation=90, labelpad=15, ha='center', va='center', fontsize=18)

    # 绘制output帧
    for t in range(T_out):
        ax = axes[2, t] if T_out > 1 else axes[2]
        im = ax.imshow(output[t, 0], cmap=vil_cmap, norm=vil_norm, vmin=vil_vmin, vmax=vil_vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if t == T_out - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if t == 0:
            ax.set_ylabel("Output", rotation=90, labelpad=15, ha='center', va='center', fontsize=18)
    plt.suptitle(f"Sample {idx}", y=0.98, fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05, wspace=0.2, hspace=0.2)
    plt.show()


def visualize_figure(model_name, dataset_name):
    """可视化为图像, 根据数据集名称调用对应的可视化函数"""
    sample_dir = f"Output/Sample/{dataset_name}/{model_name}"

    # 收集所有样本序号
    indices = set()
    for file in os.listdir(sample_dir):
        if file.endswith(".npy"):
            idx = int(file.split("_")[1].split(".")[0])
            indices.add(idx)
    indices = sorted(list(indices))

    # 根据数据集选择对应的可视化函数
    visualize_func_map = {
        'MovingMNIST': visualize_movingmnist,
        'TaxiBJ': visualize_taxibj,
        'SEVIR': visualize_sevir
    }

    if dataset_name not in visualize_func_map:
        raise ValueError(f"不支持的数据集: {dataset_name}, 支持的数据集: {list(visualize_func_map.keys())}")

    visualize_func = visualize_func_map[dataset_name]

    # 可视化每个样本
    for idx in indices:
        sample = np.load(f"{sample_dir}/{model_name}-sample_{idx}.npy",
                        allow_pickle=True).item()
        input_data = sample["input"]
        target = sample["target"]
        output = sample["output"]

        visualize_func(idx, input_data, target, output)
