import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class PatchEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, overlapping):
        super().__init__()

        kernel_size = patch_size * max(1, overlapping)
        stride = patch_size
        padding = max(0, overlapping - 1) * patch_size // 2

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.encoder(x)


class STLMixer(nn.Module):
    def __init__(self, hid_channels, k1, k2):
        super().__init__()

        # 深度卷积块, 使用了普通卷积+空洞卷积
        self.depthwise_conv = Residual(
            nn.Sequential(
                nn.Conv2d(hid_channels, hid_channels, kernel_size=k1, groups=hid_channels, padding="same"),
                nn.Conv2d(hid_channels, hid_channels, kernel_size=k2, groups=hid_channels, padding="same", dilation=3),
                nn.GELU(),
                nn.BatchNorm2d(hid_channels)
            )
        )
        
        # 点卷积块
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(hid_channels, hid_channels, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hid_channels)
        )
    
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
