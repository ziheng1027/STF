# Module/PredRNN.py
import torch
import torch.nn as nn


class STLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, height, width, kernel_size, stride, padding, use_layer_norm=True, version='V1'):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.use_layer_norm = use_layer_norm
        self.height = height
        self.width = width
        self.version = version

        # 卷积操作计算4+3个门的特征图
        self.conv_x = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels * 7,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        #  隐藏状态H的卷积(i_h, f_h, g_h, o_h)
        self.conv_h = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        #  空间记忆M的卷积
        self.conv_m = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 3,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        # 输出门卷积, 处理拼接后的记忆
        self.conv_o = nn.Conv2d(
            in_channels=hidden_channels * 2,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        # 最终输出的1x1卷积
        self.conv_out = nn.Conv2d(
            in_channels=hidden_channels * 2,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        # 层归一化
        if self.use_layer_norm:
            self.norm_x = nn.LayerNorm([hidden_channels * 7, height, width])
            self.norm_h = nn.LayerNorm([hidden_channels * 4, height, width])
            self.norm_m = nn.LayerNorm([hidden_channels * 3, height, width])
            self.norm_o = nn.LayerNorm([hidden_channels, height, width])
    
    def forward(self, x, h_prev, c_prev, m_prev):
        # 卷积运算
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h_prev)
        m_concat = self.conv_m(m_prev)

        if self.use_layer_norm:
            x_concat = self.norm_x(x_concat)
            h_concat = self.norm_h(h_concat)
            m_concat = self.norm_m(m_concat)
        
        # 拆分卷积结果
        i_x, f_x, g_x, i_x_m, f_x_m, g_x_m, o_x = torch.split(x_concat, self.hidden_channels, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.hidden_channels, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.hidden_channels, dim=1)

        # 更新时间记忆 
        i_t= torch.sigmoid(i_x + i_h)
        f_t= torch.sigmoid(f_x + f_h + 1.0)
        g_t= torch.tanh(g_x + g_h)
        c_new = f_t * c_prev + i_t * g_t
        # PredRNN-V2需要delta_c来计算解耦损失
        if self.version == 'V2':
            delta_c = i_t * g_t

        # 更新空间记忆
        i_t_m = torch.sigmoid(i_x_m + i_m)
        f_t_m = torch.sigmoid(f_x_m + f_m + 1.0)
        g_t_m = torch.tanh(g_x_m + g_m)
        m_new = f_t_m * m_prev + i_t_m * g_t_m
        # PredRNN-V2需要delta_m来计算解耦损失
        if self.version == 'V2':
            delta_m = i_t_m * g_t_m

        # 特征融合与输出
        cm_concat = torch.cat([c_new, m_new], dim=1)
        o_t = o_x + o_h + self.conv_o(cm_concat)
        if self.use_layer_norm:
            o_t = self.norm_o(o_t)
        o_t = torch.sigmoid(o_t)

        # 计算新的隐藏状态H
        h_new = o_t * torch.tanh(self.conv_out(cm_concat))

        # PredRNN-V2返回delta_c和delta_m
        if self.version == 'V2':
            return h_new, c_new, m_new, delta_c, delta_m

        return h_new, c_new, m_new