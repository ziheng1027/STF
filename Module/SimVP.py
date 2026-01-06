# Module/SimVP.py
import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, is_transpose, is_norm_act):
        super().__init__()
        self.is_norm_act = is_norm_act
        # 选择使用普通卷积还是反卷积
        if is_transpose:
            # 使用反卷积
            self.conv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=stride // 2
            )
        else:
            # 使用普通卷积
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        # 组归一化,将通道分为2组进行归一化
        self.norm = nn.GroupNorm(2, out_channels)
        # leaky_relu激活函数,斜率为0.2
        self.act = nn.LeakyReLU(0.2, inplace=True)
        # self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.is_norm_act:
            x = self.norm(x)
            x = self.act(x)
        # print("BasicConv输出形状: ", x.shape)
        return x


class SpatialConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, is_transpose, is_norm_act):
        super().__init__()
        # 如果步长为1,强制使用普通卷积(因为步长为1的反卷积和普通卷积效果相同,但计算量更大)
        if stride == 1:
            is_transpose = False
        self.conv = BasicConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            is_transpose=is_transpose,
            is_norm_act=is_norm_act
        )

    def forward(self, x):
        x = self.conv(x)
        # print("SpatialConv输出形状: ", x.shape)
        return x


class GroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, is_norm_act):
        super().__init__()
        self.is_norm_act = is_norm_act
        # 如果"通道数"不能够被"分组数"整除,则将"分组数"设置为1
        if in_channels % groups != 0:
            groups = 1
        # 分组卷积
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups
        )
        # 组归一化
        self.norm = nn.GroupNorm(groups, out_channels)
        # leaky_relu激活函数,斜率为0.2
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.is_norm_act:
            x = self.norm(x)
            x = self.act(x)
        # print("GroupConv输出形状: ", x.shape)
        return x
    

class Inception(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, inception_kernels=[3, 5, 7, 11], groups=8):
        super().__init__()
        # 1. 1x1卷积降维
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hid_channels,
            kernel_size=1,
        )
        # 2. 多尺度分组卷积
        layers = []
        for kernel_size in inception_kernels:
            layers.append(
                GroupConv(
                    in_channels=hid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    groups=groups,
                    is_norm_act=True
                )
            )
        # 将多个分组卷积模块组合成一个Sequential
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # 先进行1x1卷积降维
        x = self.conv1(x)
        # 再进行多尺度分组卷积,将所有分支的结果相加(4个卷积核对应4个分支)
        x = sum(layer(x) for layer in self.layers)
        # print("Inception输出形状: ", x.shape)
        return x


def stride_generator(num_layers, reverse=False):
    strides = [1,2] * (num_layers // 2)
    # 是否反转序列
    if reverse:
        return list(reversed(strides))
    else:
        return strides


class Encoder(nn.Module):
    def __init__(self, in_channels, hid_channels, encoder_layers):
        super().__init__()
        # 生成步长序列
        strides = stride_generator(encoder_layers)
        # 创建编码器层
        self.encoder = nn.Sequential(
            # 第一层单独处理,获取初始的输入通道,并且需要保存其输出用于跳跃连接
            SpatialConv(
                in_channels=in_channels,
                out_channels=hid_channels,
                stride=strides[0],
                is_transpose=False,
                is_norm_act=True
            ),
            # 后续层
            *[
                SpatialConv(
                    in_channels=hid_channels,
                    out_channels=hid_channels,
                    stride=stride,
                    is_transpose=False,
                    is_norm_act=True
                ) for stride in strides[1:]
            ]
        )

    def forward(self, x):
        # 保存第一层输出用于跳跃连接
        enc1 = self.encoder[0](x)
        x = enc1
        # 后续层
        for i in range(1, len(self.encoder)):
            x = self.encoder[i](x)
        # 返回最终特征以及用于跳跃连接的特征
        # print("Encoder输出形状: ", x.shape)
        return x, enc1


class Translator(nn.Module):
    def __init__(self, in_channels, hid_channels, translator_layers, kernel_sizes=[3, 5, 7, 11], groups=8):
        super().__init__()
        self.translator_layers = translator_layers
        # 创建编码器层
        encoder_layers = []
        # 第一层
        encoder_layers.append(
            Inception(
                in_channels=in_channels,
                hid_channels=hid_channels // 2,
                out_channels=hid_channels,
                inception_kernels=kernel_sizes,
                groups=groups
            )
        )
        # 中间层
        for i in range(1, translator_layers - 1):
            encoder_layers.append(
                Inception(
                    in_channels=hid_channels,
                    hid_channels=hid_channels // 2,
                    out_channels=hid_channels,
                    inception_kernels=kernel_sizes,
                    groups=groups
                )
            )
        # 最后一层
        encoder_layers.append(
            Inception(
                in_channels=hid_channels,
                hid_channels=hid_channels // 2,
                # out_channels=in_channels, # 1
                out_channels=hid_channels,  # 2 - 原文相同设置
                inception_kernels=kernel_sizes,
                groups=groups,
            )
        )
        # 创建解码器层
        decoder_layers = []
        # 第一层
        decoder_layers.append(
            Inception(
                # in_channels=in_channels, # 1
                in_channels=hid_channels,  # 2 - 原文相同设置
                hid_channels=hid_channels // 2,
                out_channels=hid_channels,
                inception_kernels=kernel_sizes,
                groups=groups
            )
        )
        # 中间层(带跳跃连接)
        for i in range(1, translator_layers - 1):
            decoder_layers.append(
                Inception(
                    in_channels=hid_channels * 2,
                    hid_channels=hid_channels // 2,
                    out_channels=hid_channels,
                    inception_kernels=kernel_sizes,
                    groups=groups
                )
            )
        # 最后一层
        decoder_layers.append(
            Inception(
                in_channels=hid_channels * 2,
                hid_channels=hid_channels // 2,
                out_channels=in_channels,
                inception_kernels=kernel_sizes,
                groups=groups,
            )
        )
        # 创建编码器和解码器
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        # 结合时间和通道维度进行卷积
        x = x.reshape(B, T*C, H, W)
        # 编码过程
        skips = []
        for i in range(self.translator_layers):
            x = self.encoder[i](x)
            if i < self.translator_layers - 1:
                skips.append(x)
        # 解码过程
        x = self.decoder[0](x)
        for i in range(1, self.translator_layers):
            x = torch.cat([x, skips[-i]], dim=1)
            x = self.decoder[i](x)
        # 再拆分时间和通道维度
        x = x.reshape(B, T, C, H, W)
        # print("Translator输出形状: ", x.shape)
        return x
    

class Decoder(nn.Module):
    def __init__(self, hid_channels, out_channels, decoder_layers):
        super().__init__()
        # 获取反转的步长序列
        strides = stride_generator(decoder_layers, reverse=True)
        # 创建解码器层
        self.decoder = nn.Sequential(
            *[
                SpatialConv(
                    in_channels=hid_channels,
                    out_channels=hid_channels,
                    stride=stride,
                    is_transpose=True,
                    is_norm_act=True
                ) for stride in strides[:-1]
            ],
            # 最后一层处理跳跃连接
            SpatialConv(
                in_channels=hid_channels * 2,
                out_channels=hid_channels,
                stride=strides[-1],
                is_transpose=True,
                is_norm_act=True
            )
        )
        # 最终输出层1x1卷积
        self.out = nn.Conv2d(
            in_channels=hid_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x, enc1=None):
        # 逐层解码
        for i in range(len(self.decoder) - 1):
            x = self.decoder[i](x)
        # 最后一层处理跳跃连接
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder[-1](x)
        # 最终输出层1x1卷积
        x = self.out(x)
        # print("Decoder输出形状: ", x.shape)
        return x