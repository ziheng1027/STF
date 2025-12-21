import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



class SEVIRDataset(Dataset):
    def __init__(self, data_path, type='vil'):
        self.data_path = data_path

        with h5py.File(self.data_path, 'r') as f:
            self.vil = f[type][:]

        print(f"SEVIR dataset loaded. Shape: {self.vil.shape}")

    def __len__(self):
        return len(self.vil)

    def __getitem__(self, idx):
        full_seq = self.vil[idx]
        inputs = full_seq[0:7]
        target = full_seq[7:13]
        
        inputs_tensor = torch.from_numpy(inputs.astype(np.float32) / 255.0)
        target_tensor = torch.from_numpy(target.astype(np.float32) / 255.0)

        return inputs_tensor, target_tensor

def get_dataloader(train_data_path, test_data_path, batch_size_train, batch_size_valid, valid_ratio=0.1, num_workers=4):
    train_dataset = SEVIRDataset(train_data_path)
    test_dataset = SEVIRDataset(test_data_path)

    total_train_size = len(train_dataset)
    val_size = int(total_train_size * valid_ratio)
    train_size = total_train_size - val_size

    print(f"Total training size: {total_train_size}, for train: {train_size}, for valid: {val_size}")
    
    # 随机划分训练集和验证集
    train_subset, valid_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_subset,
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
        train_path, test_path, batch_size=16, val_ratio=0.1, num_workers=0
    )

    import matplotlib.pyplot as plt
    for inputs, targets in train_loader:
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
            
