# Dataset/SEVIR.py
import os
import h5py
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



class TransformsFixRotation(nn.Module):
    r"""
    Rotate by one of the given angles.
    """
    def __init__(self, angles):
        super().__init__()
        if not isinstance(angles, (list, tuple)):
            angles = [angles, ]
        self.angles = angles

    def forward(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angles={self.angles})"


class SEVIRDataset(Dataset):
    def __init__(self, data_path, type='vil', is_train=False):
        self.data_path = data_path
        self.is_train = is_train

        with h5py.File(self.data_path, 'r') as f:
            self.vil = f[type][:]

        print(f"SEVIR dataset loaded. Shape: {self.vil.shape}")

        if self.is_train:
            self.augment = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                TransformsFixRotation(angles=[0, 90, 180, 270])
            )


    def __len__(self):
        return len(self.vil)

    def __getitem__(self, idx):
        full_seq = self.vil[idx]
        full_seq_tensor = torch.from_numpy(full_seq.astype(np.float32) / 255.0)
        if self.is_train:
            full_seq_tensor = self.augment(full_seq_tensor)

        inputs = full_seq_tensor[0:7]
        target = full_seq_tensor[7:13]

        return inputs, target

def get_dataloader(train_data_path, test_data_path, batch_size_train, batch_size_valid, valid_ratio=0.1, num_workers=4):
    train_dataset = SEVIRDataset(train_data_path, type='vil', is_train=True)
    test_dataset = SEVIRDataset(test_data_path, type='vil', is_train=False)

    total_test_size = len(test_dataset)
    valid_size = int(total_test_size * valid_ratio)
    test_size = total_test_size - valid_size

    print(f"Total test size: {total_test_size}, for valid: {valid_size}, for test: {test_size}")
    
    # 随机划分验证集和测试集
    valid_dataset, test_dataset = random_split(
        test_dataset,
        [valid_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size_valid,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_valid,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    train_path = r"Data\SEVIR\SEVIR_VIL_STORMEVENTS_train.h5"
    test_path = r"Data\SEVIR\SEVIR_VIL_STORMEVENTS_test.h5"

    train_loader, valid_loader, test_loader = get_dataloader(
        train_path, test_path, batch_size_train=16, batch_size_valid=4, valid_ratio=0.1, num_workers=0
    )

    import matplotlib.pyplot as plt
    for inputs, targets in test_loader:
        # inputs shape: (16, 7, 128, 128) targets shape: (16, 6, 128, 128)
        B= inputs.size(0)
        for b in range(B):
            # 绘制2行6列-6帧输入(排除第一帧取后6帧), 6帧输出
            plt.figure(figsize=(12, 6))
            for t in range(1, inputs.shape[1]):
                plt.subplot(2, 6, t)
                plt.imshow(inputs[b, t, 0, :, :])
                plt.title(f'Input Frame {t}')
                plt.axis('off')
            for t in range(targets.shape[1]):
                plt.subplot(2, 6, t + 6 + 1)
                plt.imshow(targets[b, t, 0, :, :])
                plt.title(f'Target Frame {t+1}')
                plt.axis('off')
            plt.tight_layout()
            plt.show()
            
