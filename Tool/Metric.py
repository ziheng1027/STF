import torch
import numpy as np


def MSE(target, pred):
    return np.mean(np.square(target - pred), axis=(0, 1)).sum()

def MAE(target, pred):
    return np.mean(np.abs(target - pred), axis=(0, 1)).sum()

def PSNR(target, pred):
    mse = MSE(target, pred)
    if mse == 0:
        return 100
    max_pixel = 1.0 if target.max() <= 1 else 255
    return 20 * np.log10(max_pixel / np.sqrt(mse))