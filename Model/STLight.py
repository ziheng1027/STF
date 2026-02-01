import torch
import torch.nn as nn
from Module.STLight import PatchEncoder, STLMixer


class STLight(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, layers, patch_size, overlapping, k1, k2):
        super().__init__()
        
        self.out_channels = out_channels
        self.patch_size = patch_size

        # patch编码器
        self.encoder = PatchEncoder(in_channels, hid_channels, patch_size, overlapping)
        # patch后时序处理
        self.translator = nn.ModuleList([
            STLMixer(hid_channels, k1, k2) for _ in range(layers)
        ])
        # patch -> 原始图片分辨率
        self.up = nn.PixelShuffle(patch_size)
        # 通道数hid_channels -> PixelShuffle后的hid_channels -> out_channels(T*C)
        self.patch_reassemble = nn.Conv2d(hid_channels // (patch_size ** 2), out_channels, kernel_size=1)
        # 初始化权重
        self.encoder.apply(self._init_weights)
        self.translator.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        # patch编码
        x = self.encoder(x)
        # 采用时序处理
        x1 = None
        layers_T = len(self.translator)

        for i, block in enumerate(self.translator):
            # 在1/3处缓存特征
            if i == layers_T // 3:
                x1 = x
            # 在2/3处进行残差连接
            if i == 2 * layers_T // 3:
                x = x + x1

            x = block(x)

        # patch恢复原分辨率
        x = self.up(x)
        x = self.patch_reassemble(x)
        # 拆分T*C输出
        x = x.reshape(B, self.out_channels // C, C, H, W)
        return x