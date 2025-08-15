import torch.nn as nn
from Module.UNet import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, out_frames, bilinear=False):
        """
        初始化U-Net
        
        参数:
            n_channels (int): 输入通道数
            n_classes (int): 输出类别数
            bilinear (bool): 是否使用双线性插值进行上采样
        """
        super().__init__()

        self.n_channels = in_channels  # 输入图像的通道数
        self.n_classes = out_channels  # 输出的通道数
        self.out_frames = out_frames   # 输出的帧数
        self.bilinear = bilinear       # 上采样方式
        
        # 初始的双重卷积
        self.inconv = DoubleConv(in_channels, 64)
        
        # 下采样路径（编码器）
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # 如果使用双线性插值，最后一层通道数减半
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # 上采样路径（解码器）
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 输出层
        self.outconv = OutConv(64, out_channels * out_frames)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        # 编码器部分
        x1 = self.inconv(x)       # 输入特征
        x2 = self.down1(x1)       # 第一次下采样
        x3 = self.down2(x2)       # 第二次下采样
        x4 = self.down3(x3)       # 第三次下采样
        x5 = self.down4(x4)       # 第四次下采样（瓶颈层）
        
        # 解码器部分（带跳跃连接）
        x = self.up1(x5, x4)      # 第一次上采样，连接x4
        x = self.up2(x, x3)       # 第二次上采样，连接x3
        x = self.up3(x, x2)       # 第三次上采样，连接x2
        x = self.up4(x, x1)       # 第四次上采样，连接x1
        
        # 输出层
        output = self.outconv(x)     # 最终输出

        # 重塑为[B, T, C, H, W]
        output = output.reshape(B, self.out_frames, C, H, W)
        return output