import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Tool.Utils import worker_init_fn
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class TaxiBJDataset(Dataset):
    def __init__(self, data_path, is_train, use_augment=False):
        """
        TaxiBJ Dataset
        
        Args:
            data_path: 数据文件路径
            is_train: 是否为训练集
            use_augment: 是否使用数据增强
        """
        super().__init__()
        self.is_train = is_train
        self.use_augment = use_augment

        data = np.load(data_path)
        if is_train:
            self.inputs = data['X_train']
            self.targets = data['Y_train']
            # 从(-1,1)映射到(0,1)
            self.inputs = (self.inputs + 1) / 2
            self.targets = (self.targets + 1) / 2
        else:
            self.inputs = data['X_test']
            self.targets = data['Y_test']
            # 从(-1,1)映射到(0,1)
            self.inputs = (self.inputs + 1) / 2
            self.targets = (self.targets + 1) / 2
    
    def augment_seq(self, taxibj_seq):
        """数据增强"""
        if random.randint(0, 1):
            # 50%的概率进行翻转
            taxibj_seq = torch.flip(taxibj_seq, dims=(3, ))
        return taxibj_seq
    
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        inputs = torch.tensor(self.inputs[idx, ::]).float()
        targets = torch.tensor(self.targets[idx, ::]).float()
        if self.use_augment:
            data_length = inputs.shape[0]
            # 拼接输入和输出, 一起进行数据增强
            taxibj_seq = self.augment_seq(torch.cat([inputs, targets], dim=0))
            inputs = taxibj_seq[:data_length, ...]
            targets = taxibj_seq[data_length:, ...]
        return inputs, targets

def get_dataloader(data_path, use_augment, batch_size_train, batch_size_valid, num_workers=4):
    """获取数据加载器"""
    train_dataset = TaxiBJDataset(
        data_path=data_path,
        is_train=True,
        use_augment=use_augment
    )
    test_dataset = TaxiBJDataset(
        data_path=data_path,
        is_train=False,
        use_augment=False
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    valid_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_valid,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_valid,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = get_dataloader(
        data_path = r'Data\TaxiBJ\taxibj.npz',
        use_augment=True,
        batch_size_train=16,
        batch_size_val=4,
        num_workers=0
    )
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    for inputs, targets in train_loader:
        print("Inputs shape:", inputs.shape)
        print("Targets shape:", targets.shape)
        B, T = inputs.shape[0], inputs.shape[1]
        for b in range(B):
            plt.figure(figsize=(10, 6))
            for t in range(T):
                plt.subplot(2, T, t+1)
                plt.axis('off')
                plt.imshow(inputs[b, t, 0])
            for t in range(T):
                plt.subplot(2, T, t+1+T)
                plt.axis('off')
                plt.imshow(targets[b, t, 0])
            plt.tight_layout()
            plt.show()