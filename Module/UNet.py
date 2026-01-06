# Module/UNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    '''双重卷积模块: (Conv => BN => ReLU) x 2'''
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        # 如果没指定中间通道数，则使用输出通道数
        if not mid_channels:
            mid_channels = out_channels
        
        # 构建双重卷积
        self.double_conv = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 第二个卷积块
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    '''下采样模块: 最大池化 + 双重卷积'''
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 构建下采样模块
        self.maxpool_conv = nn.Sequential(
            # 2x2最大池化, 步长为2
            nn.MaxPool2d(2),
            # 接双重卷积
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    '''上采样模块: 上采样 + 双重卷积'''
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 根据参数选择使用双线性插值或转置卷积进行上采样
        if bilinear:
            # 使用双线性插值进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 因为使用了跳跃连接，所以输入通道数为in_channels // 2
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 使用转置卷积进行上采样
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        # x1是来自上一层的输入, x2是跳跃连接的输入

        # 对x1进行上采样
        x1 = self.up(x1)

        # 计算x1和x2的尺寸差异（input格式是C, H, W）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 对x1进行padding，使其与x2尺寸匹配
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # 将x1和x2在通道维度上拼接
        x = torch.cat([x2, x1], dim=1)

        # 将拼接结果输入到卷积层
        return self.conv(x)
    

class OutConv(nn.Module):
    """输出卷积层: 1x1卷积, 将通道数映射到类别数"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)