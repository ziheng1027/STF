import torch.nn as nn
from Module.SmaAtUNet import DoubleConvDS, DownDS, UpDS, OutConv
from Module.Attention.CBAM import CBAM


class SmaAt_UNet(nn.Module):
    def __init__(self, in_channels, out_channels, out_frames=10, bilinear=True, num_kernel=2, reduction_ratio=16):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_frames = out_frames
        self.num_kernel = num_kernel
        self.bilinear = bilinear
        self.reduction_ratio = reduction_ratio

        # 编码器
        self.inconv = DoubleConvDS(self.in_channels, 64, num_kernel=num_kernel)
        self.cbam1 = CBAM(64, reduction_ratio)
        self.down1 = DownDS(64, 128, num_kernel=num_kernel)
        self.cbam2 = CBAM(128, reduction_ratio)
        self.down2 = DownDS(128, 256, num_kernel=num_kernel)
        self.cbam3 = CBAM(256, reduction_ratio)
        self.down3 = DownDS(256, 512, num_kernel=num_kernel)
        self.cbam4 = CBAM(512, reduction_ratio)
        factor = 2 if bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, num_kernel=num_kernel)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio)

        # 解码器
        self.up1 = UpDS(1024, 512 // factor, bilinear, num_kernel=num_kernel)
        self.up2 = UpDS(512, 256 // factor, bilinear, num_kernel=num_kernel)
        self.up3 = UpDS(256, 128 // factor, bilinear, num_kernel=num_kernel)
        self.up4 = UpDS(128, 64, bilinear, num_kernel=num_kernel)
        self.outconv = OutConv(64, out_channels * out_frames)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T*C, H, W)
        # 编码
        x1 = self.inconv(x)
        x1 = self.cbam1(x1)
        x2 = self.down1(x1)
        x2 = self.cbam2(x2)
        x3 = self.down2(x2)
        x3 = self.cbam3(x3)
        x4 = self.down3(x3)
        x4 = self.cbam4(x4)
        x5 = self.down4(x4)
        x5 = self.cbam5(x5)
        # 解码
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outconv(x)
        # 重塑为[B, T, C, H, W]
        out = out.reshape(B, self.out_frames, C, H, W)
        return out