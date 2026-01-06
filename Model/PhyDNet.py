# Model/PhyDNet.py
import torch
import torch.nn as nn
from Module.PhyDNet import Encoder, Decoder, PREncoder, PRDecoder


class PhyDNet(nn.Module):
    def __init__(self, phycell, convlstm, device):
        """
        参数:
            phycell:物理约束网络实例
            convcell:卷积LSTM网络实例
            device:计算设备
        """
        super().__init__()
        self.encoder = Encoder().to(device)        # 通用编码器

        # 1. PhyCell分支
        self.phy_encoder = PREncoder().to(device)  # PhyCell分支编码器
        self.phycell = phycell.to(device)          # PhyCell网络
        self.phy_decoder = PRDecoder().to(device)  # PhyCell分支解码器
        # 2. ConvLSTM分支
        self.res_encoder = PREncoder().to(device)  # ConvLSTM分支编码器
        self.convlstm = convlstm.to(device)        # ConvLSTM网络
        self.res_decoder = PRDecoder().to(device)  # ConvLSTM分支解码器

        self.decoder = Decoder().to(device)        # 通用解码器

    def forward(self, input, first_timestep=False, decoding=False):
        """
        Args:
            input: 输入图像张量
            first_timestep: 是否是该序列的第一个时间步
            decoding: 是否处于解码阶段
            
        Returns:
            (output, phy_output, hidden_phy, res_output): 
            - 最终输出图像
            - 物理分支重建结果
            - 物理分支隐藏状态
            - 残差分支重建结果
        """
        # 通用编码器
        input = self.encoder(input)
        if decoding:
            input_phy = None
        else:
            # PhyCell分支编码器
            input_phy = self.phy_encoder(input)
        # ConvLSTM分支编码器
        input_res = self.res_encoder(input)

        # PhyCell分支
        hidden_phy, output_phy = self.phycell(input_phy, first_timestep)
        # ConvLSTM分支
        hidden_res, output_res = self.convlstm(input_res, first_timestep)

        # PhyCell分支解码器
        decoded_phy = self.phy_decoder(output_phy[-1])
        # ConvLSTM分支解码器
        decoded_res = self.res_decoder(output_res[-1])

        # 部分重建结果用于可视化
        phy_output = torch.sigmoid(self.decoder(decoded_phy))
        res_output = torch.sigmoid(self.decoder(decoded_res))

        # 合并两个分支的结果
        concat = decoded_phy + decoded_res
        output = torch.sigmoid(self.decoder(concat))

        return output, phy_output, res_output, hidden_phy