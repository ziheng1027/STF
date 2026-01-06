# Model/SimVP.py
import torch.nn as nn
from Module.SimVP import Encoder, Translator, Decoder


class SimVP(nn.Module):
    def __init__(self, input_shape, hid_channels_S=64, hid_channels_T=256, 
                 layers_S=4, layers_T=8, inception_kernels=[3, 5, 7, 11], groups=8):
            super().__init__()

            T, C, H, W = input_shape

            # 初始化空间编码器,时序转换器,空间解码器
            self.encoder = Encoder(
                in_channels=C,
                hid_channels=hid_channels_S,
                encoder_layers=layers_S
            )
            self.translator = Translator(
                in_channels=T * hid_channels_S,
                hid_channels=hid_channels_T,
                translator_layers=layers_T,
                kernel_sizes=inception_kernels,
                groups=groups
            )
            self.decoder = Decoder(
                hid_channels=hid_channels_S,
                out_channels=C,
                decoder_layers=layers_S
            )

    def forward(self, x):
        B, T, C, H, W = x.shape
        # 1. 重置输入用于空间编码
        # 将B和T相乘后,新的批处理大小变为B * T,这意味着每个时间步长的每一帧都被视为一个独立的样本。
        x = x.view(B * T, C, H, W)
        # 2. 空间编码
        x, skip = self.encoder(x)
        _, C_hid, H_hid, W_hid = x.shape
        # 3. 重塑输入用于时序转换
        x = x.view(B, T, C_hid, H_hid, W_hid) 
        # 4. 时序特征提取
        x = self.translator(x)
        x = x.reshape(B*T, C_hid, H_hid, W_hid)
        # 5. 空间解码(使用跳跃连接)
        x = self.decoder(x, skip)
        x = x.view(B, T, C, H, W)

        return x