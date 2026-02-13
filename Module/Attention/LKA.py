# Module/Attention/LKA.py
import math
import torch
import torch.nn as nn


class LKA(nn.Module):
    def __init__(self, hid_channels):
        """基础LKA注意力算子"""
        super().__init__()

        self.conv0 = nn.Conv2d(
            hid_channels, 
            hid_channels, 
            kernel_size=5, 
            padding=2, 
            groups=hid_channels
        )
        self.conv_spatial = nn.Conv2d(
            hid_channels, 
            hid_channels, 
            kernel_size=7, 
            stride=1, 
            padding=9, 
            groups=hid_channels, 
            dilation=3
        )
        self.conv1 = nn.Conv2d(hid_channels, hid_channels, 1)

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class LargeKernelAttentionModule(nn.Module):
    """gSTA使用的LKA算子(LKA+split门控)"""
    def __init__(self, hid_channels, kernel_size=21, dilation=3):
        super().__init__()

        # 计算拆解后的卷积核大小和填充
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        # 局部特征提取层-DW Conv
        self.conv0 = nn.Conv2d(
            hid_channels,
            hid_channels,
            kernel_size=d_k,
            padding=d_p,
            groups=hid_channels
        )
        # 长程依赖提取层-DW-D Conv
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)
        self.conv_spatial = nn.Conv2d(
            hid_channels,
            hid_channels,
            kernel_size=dd_k,
            padding=dd_p,
            groups=hid_channels,
            dilation=dilation
        )
        # 通道交互与门控生成层, 输出*2用于后续split
        self.conv1 = nn.Conv2d(hid_channels, 2 * hid_channels, 1)
    
    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn) # C = 2 * hid_channels

        # 门控: 将输出平分为两部分, 一部分作为特征, 一部分作为权重
        f_x, g_x = torch.split(attn, attn.shape[1] // 2, dim=1)
        # 使用Sigmoid激活权重分支并与特征分支相乘
        return torch.sigmoid(g_x) * f_x


class SpatialGatingAttention(nn.Module):
    """gSTA使用的空间门控组件"""
    def __init__(self, hid_channels, kernel_size=21, attn_shortcut=True):
        super().__init__()

        # 投影层1-将输入映射到注意力计算空间
        self.proj1 = nn.Conv2d(hid_channels, hid_channels, kernel_size=1)
        self.act_func = nn.GELU()
        # 大核注意力
        self.spatial_gating_unit = LargeKernelAttentionModule(hid_channels, kernel_size)
        # 投影层2-融合计算后的特征
        self.proj2 = nn.Conv2d(hid_channels, hid_channels, kernel_size=1)
        self.attn_shortcut = attn_shortcut
    
    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        
        x = self.proj1(x)
        x = self.act_func(x)
        x = self.spatial_gating_unit(x)
        x = self.proj2(x)

        # 残差连接
        if self.attn_shortcut:
            x = x + shortcut
        return x


class TemporalAttentionModule(nn.Module):
    """TAU使用的时间增强注意力算子(LKA + SE)"""
    def __init__(self, hid_channels, kernel_size=21, dilation=3, reduction=16):
        super().__init__()

        # 基础LKA卷积拆解
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        self.conv0 = nn.Conv2d(
            hid_channels,
            hid_channels,
            kernel_size=d_k,
            padding=d_p,
            groups=hid_channels
        )
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)
        self.conv_spatial = nn.Conv2d(
            hid_channels,
            hid_channels,
            kernel_size=dd_k,
            padding=dd_p,
            groups=hid_channels,
            dilation=dilation
        )

        # 此处输出维度为dim, 不再split
        self.conv1 = nn.Conv2d(hid_channels, hid_channels, kernel_size=1)
        # SE通道注意力模块: 捕捉不同时间步/通道的重要性
        reduction = max(hid_channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hid_channels, hid_channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(hid_channels // reduction, hid_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 原始输入u用于最终加权
        u = x.clone()
        # 提取空间维度的特征投影f_x
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        f_x = self.conv1(attn)
        # 计算通道维度的权重
        B, C, _, _ = x.size()
        attn_se = self.avg_pool(x).view(B, C)
        attn_se = self.fc(attn_se).view(B, C, 1, 1)
        # 加权融合
        return attn_se * f_x * u


class TemporalAttention(nn.Module):
    """TAU使用的时间注意力组件"""
    def __init__(self, hid_channels, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj1 = nn.Conv2d(hid_channels, hid_channels, kernel_size=1)
        self.act_func = nn.GELU()
        self.spatial_gating_unit = TemporalAttentionModule(hid_channels, kernel_size)
        self.proj2 = nn.Conv2d(hid_channels, hid_channels, kernel_size=1)
        self.attn_shortcut = attn_shortcut
    
    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        
        x = self.proj1(x)
        x = self.act_func(x)
        x = self.spatial_gating_unit(x)
        x= self.proj2(x)

        if self.attn_shortcut:
            x = x + shortcut
        return x

def init_weights(m):
    """权重初始化函数"""
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
        # 使用 PyTorch 原生的正态分布初始化线性层
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)