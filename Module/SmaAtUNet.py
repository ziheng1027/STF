import torch
import torch.nn.functional as F
from torch import nn


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, num_kernel=1):
        super().__init__()

        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * num_kernel,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels * num_kernel,
            out_channels=out_channels,
            kernel_size=1
        )
    
    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class DoubleConvDS(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels=None, num_kernel=1):
        super().__init__()

        if not hid_channels: hid_channels = out_channels
        self.doubel_conv = nn.Sequential(
            DSConv(in_channels, hid_channels, 3, 1, num_kernel),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            DSConv(hid_channels, out_channels, 3, 1, num_kernel),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out = self.doubel_conv(x)
        return out


class DownDS(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernel=1):
        super().__init__()

        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(in_channels, out_channels, num_kernel=num_kernel)
        )
    
    def forward(self, x):
        out = self.max_pool_conv(x)
        return out


class UpDS(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, num_kernel=1):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS(in_channels, out_channels, in_channels//2, num_kernel=num_kernel)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, num_kernel=num_kernel)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        return out