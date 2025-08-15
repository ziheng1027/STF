import os
import gzip
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class MovingMNISTDataset(Dataset):
    def __init__(self, mnist_path, test_dataset_path, is_train, input_frames=10, 
                 output_frames=10, image_size=64, num_digits=[2], use_augment=False):
        """
        Moving MNIST Dataset

        Args:
            mnist_path (str): 数据文件路径
            is_train (bool): 是否为训练集
            input_frames (int): 输入帧数
            output_frames (int): 输出帧数
            image_size (int): 图像大小,默认为64
            num_digits (list): 图像中移动数字的数量,默认为[2],可扩展为[2,3,4]等
            use_augment (bool): 是否使用数据增强,默认为False
        """
        super().__init__()
        # 设置参数
        self.is_train = is_train
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.num_digits = num_digits
        self.image_size = image_size
        self.use_augment = use_augment

        # mnist数据集图像尺寸28
        self.mnist_size = 28

        # 数字移动步长
        self.step_length = 0.1
        
        # 数据集(用于加载固定测试集)
        self.dataset = None

        # 加载数据集
        if self.is_train:
            self.mnist = self.load_mnist(mnist_path)
        else:
            # 动态随机生成测试集
            if len(num_digits) != 1:
                self.mnist = self.load_mnist(mnist_path)
            # 加载固定测试集
            else:
                self.dataset = self.load_fixed_set(test_dataset_path)
        
        # 设置数据集长度
        self.length = int(10000) if self.dataset is None else self.dataset.shape[1]

    def load_mnist(self, data_path):
        """加载mnist数据集"""
        with gzip.open(data_path, 'rb') as f:
            mnist = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            mnist = mnist.reshape(-1, 28, 28)

        return mnist
    
    def load_fixed_set(self, data_path):
        """加载固定测试集"""
        dataset = np.load(data_path)
        dataset = dataset[..., np.newaxis]

        return dataset
    
    def get_random_trajectory(self, seq_length):
        """
        生成随机轨迹
        Args:
            seq_length (int): 序列长度
        return:
            start_x, start_y: 轨迹点的x,y坐标
        """
        # 画布大小
        canvas_size = self.image_size - self.mnist_size
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi

        # 每一帧的移动速度(周期变化)
        v_xs = [np.cos(theta)] * seq_length
        v_ys = [np.sin(theta)] * seq_length
        
        start_x = np.zeros(seq_length)
        start_y = np.zeros(seq_length)

        # 移动边界
        bound_x = 1
        bound_y = 1

        for i, v_x, v_y in zip(range(seq_length), v_xs, v_ys):
            # 沿着速度方向移动
            x += v_x * self.step_length * bound_x
            y += v_y * self.step_length * bound_y

            # 边界检测+反弹
            if x <= 0:
                x = 0
                bound_x = -bound_x
            if x >= 1:
                x = 1
                bound_x = -bound_x
            if y <= 0:
                y = 0
                bound_y = -bound_y
            if y >= 1:
                y = 1
                bound_y = -bound_y
            start_x[i] = x
            start_y[i] = y
        
        # 缩放轨迹坐标到画布大小内
        start_x = (start_x * canvas_size).astype(np.int32)
        start_y = (start_y * canvas_size).astype(np.int32)

        return start_x, start_y

    def generate_moving_mnist(self, num_digits=2):
        """
        生成随机MovingMNIST序列

        Args:
            num_digits (int): 移动数字的数量
        return:
            data_seq: 随机MovingMNIST序列
        """
        # 初始化黑色背景, (20, 64, 64)
        mmnist_seq = np.zeros(
            (self.input_frames + self.output_frames, self.image_size, self.image_size), dtype=np.float32
        )

        # 为每个数字生成轨迹图像
        for n in range(num_digits):
            # 获取随机轨迹
            start_x, start_y = self.get_random_trajectory(self.input_frames + self.output_frames)
            # 随机选择一个数字
            idx = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[idx].copy()

            # 绘制每一帧
            for i in range(self.input_frames + self.output_frames):
                # 计算数字在当前帧中的位置
                left = start_x[i]
                top = start_y[i]
                right = left + self.mnist_size
                bottom = top + self.mnist_size
            
                mmnist_seq[i, top:bottom, left:right] = np.maximum(
                    mmnist_seq[i, top:bottom, left:right], digit_image
                )
        
        # 添加通道维度
        mmnist_seq = np.expand_dims(mmnist_seq, axis=1)

        return mmnist_seq
    
    def augment_seq(self, mmnist_seq, crop_scale=0.94):
        """
        对图像序列进行数据增强

        Args:
            mmnist_seq (np.ndarray): 图像序列
            crop_scale (float): 裁剪比例
        return:
            mmnist_seq: 数据增强后的图像序列
        """
        # 将 NumPy 数组转换为 PyTorch 张量
        mmnist_seq = torch.from_numpy(mmnist_seq).float()

        # [T, C, H, W]
        _, _, h, w = mmnist_seq.shape

        mmnist_seq = F.interpolate(mmnist_seq, scale_factor=1 / crop_scale, mode='bilinear')

        _, _, h_, w_ = mmnist_seq.shape

        # 随机裁剪
        x = random.randint(0, h_ - h + 1)
        y = random.randint(0, w_ - w + 1)            
        mmnist_seq = mmnist_seq[:, :, x:x + h, y:y + w]

        # 随机翻转
        if random.randint(-2, 1):
            # 反转
            mmnist_seq = torch.flip(mmnist_seq, dims=(2, 3))
        elif random.randint(-2, 1):
            # 垂直翻转
            mmnist_seq = torch.flip(mmnist_seq, dims=(2,))
        elif random.randint(-2, 1):
            # 水平翻转
            mmnist_seq = torch.flip(mmnist_seq, dims=(3,))
        # 确保尺寸一致
        mmnist_seq = F.interpolate(mmnist_seq, size=(h, w), mode='bilinear')

        return mmnist_seq
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        seq_length = self.input_frames + self.output_frames

        if self.is_train or len(self.num_digits) != 1:
            # 随机确定图像中移动的数字数量
            num_digits = random.choice(self.num_digits)
            # 动态生成数据
            mmnist_seq = self.generate_moving_mnist(num_digits)
        else:
            # 使用固定测试集
            mmnist_seq = self.dataset[:, idx, :, :]
        
        r, w = 1, self.image_size
        mmnist_seq = mmnist_seq.reshape(seq_length, w, r, w, r).transpose(0, 2, 4, 1, 3).reshape(seq_length, r * r, w, w)
        
        # 是否数据增强
        if self.use_augment:
            mmnist_seq = self.augment_seq(mmnist_seq)
        
        # 输入序列
        inputs = mmnist_seq[:self.input_frames]
        # 标签序列
        targets = mmnist_seq[self.input_frames:seq_length]

        # 归一化
        if self.use_augment:
            inputs = inputs / 255.0
            targets = targets / 255.0
        else:
            inputs = torch.from_numpy(inputs / 255.0).contiguous().float()
            targets = torch.from_numpy(targets / 255.0).contiguous().float()

        return inputs, targets
    
def get_dataloader(data_path, input_frames, output_frames, use_augment, 
                   batch_size_train, batch_size_val, num_workers=4):
    """获取数据加载器"""
    train_set = MovingMNISTDataset(
        mnist_path=data_path,
        test_dataset_path=r"Data\MovingMNIST\mnist_test_seq.npy",
        is_train=True,
        input_frames=input_frames,
        output_frames=output_frames,
        image_size=64,
        num_digits=[2],
        use_augment=use_augment
    )
    test_set = MovingMNISTDataset(
        mnist_path=data_path,
        test_dataset_path=r"Data\MovingMNIST\mnist_test_seq.npy",
        is_train=False,
        input_frames=input_frames,
        output_frames=output_frames,
        image_size=64,
        num_digits=[2],
        use_augment=False
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloader(
        data_path=r"Data\MovingMNIST\train-images-idx3-ubyte.gz",
        input_frames=10,
        output_frames=10,
        use_augment=False,
        batch_size_train=16,
        batch_size_val=4,
        num_workers=4
    )
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    for inputs, targets in train_loader:
        print("Inputs shape:", inputs.shape)
        print("Targets shape:", targets.shape)
        B, T = inputs.shape[0], inputs.shape[1]
        for b in range(B):
            plt.figure(figsize=(16, 5))
            for t in range(T):
                plt.subplot(2, T, t+1)
                plt.imshow(inputs[b, t, 0])
                plt.axis('off')
            for t in range(T):
                plt.subplot(2, T, t+1+T)
                plt.imshow(targets[b, t, 0])
                plt.axis('off')
            plt.tight_layout()
            plt.show()