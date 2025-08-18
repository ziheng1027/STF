import torch
import numpy as np


def cal_mse(target, pred):
    return np.mean(np.square(target - pred), axis=(0, 1)).sum()

def cal_mae(target, pred):
    return np.mean(np.abs(target - pred), axis=(0, 1)).sum()

def cal_psnr(target, pred):
    mse = cal_mse(target, pred)
    if mse == 0:
        return 100
    max_pixel = 1.0 if target.max() <= 1 else 255
    return 20 * np.log10(max_pixel / np.sqrt(mse))



def cal_metrics(target, pred):
    return {
        "mse": cal_mse(target, pred),
        "mae": cal_mae(target, pred),
        "psnr": cal_psnr(target, pred)
    }