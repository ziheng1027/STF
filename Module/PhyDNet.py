# Module/PhyDNet.py
import torch
import torch.nn as nn
import numpy as np
from scipy.special import factorial
from functools import reduce


class PhyCellBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        # 计算偏导数
        self.phyconv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=(1,1),
                padding=self.padding
            ),

            nn.GroupNorm(7, hidden_dim),

            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=input_dim,
                kernel_size=(1,1),
                stride=(1,1),
                padding=(0,0)
            )
        )

        # 门控修正机制
        self.convgate = nn.Conv2d(
            in_channels=self.input_dim + self.input_dim,
            out_channels=self.input_dim,
            kernel_size=(3,3),
            padding=(1,1),
            bias=self.bias
        )
    
    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], dim=1)
        combined_conv = self.convgate(combined)
        # 计算修正系数K
        K = torch.sigmoid(combined_conv)
        # 预测阶段
        hidden_pred = hidden + self.phyconv(hidden)
        # 修正阶段
        next_hidden = hidden_pred + K * (x - hidden_pred)
        return next_hidden
    

class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, n_layers, kernel_size, device):
        super().__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []
        self.device = device

        cell_list = []
        for i in range(self.n_layers):
            cell_list.append(
                PhyCellBlock(
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, first_timestep=False):
        batch_size = input.size(0)
        # 每个序列开始时初始化隐藏状态
        if first_timestep:
            self.initHidden(batch_size)

        for i, cell in enumerate(self.cell_list):
            # 底层处理原始输入
            if i==0:
                self.H[i] = cell(input, self.H[i])
            # 上层处理前一层的输出
            else:
                self.H[i] = cell(self.H[i-1], self.H[i])
        # 返回隐藏状态和输出(此处两者相同)
        return self.H, self.H

    def initHidden(self, batch_size):
        self.H = []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]).to(self.device)
            )
    
    def setHidden(self, H):
        self.H = H


class ConvLSTMBlock(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        super().__init__()
        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, x, hidden):
        h_cur, c_cur = hidden

        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    

class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, n_layers, kernel_size, device):
        super().__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [], []
        self.device = device

        cell_list = []
        for i in range(self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(
                ConvLSTMBlock(
                    input_shape=self.input_shape,
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, first_timestep=False):
        batch_size = input.size(0)
        if first_timestep:
            self.initHidden(batch_size)

        for i, cell in enumerate(self.cell_list):
            # 底层处理原始输入
            if i==0:
                self.H[i], self.C[i] = cell(input, (self.H[i], self.C[i]))
            # 上层处理前一层的输出
            else:
                self.H[i], self.C[i] = cell(self.H[i-1], (self.H[i], self.C[i]))
        # 返回隐藏状态和输出
        return (self.H, self.C), self.H

    def initHidden(self, batch_size):
        self.H, self.C = [], []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.hidden_dim[i], self.input_shape[0], self.input_shape[1]).to(self.device)
            )
            self.C.append(
                torch.zeros(batch_size, self.hidden_dim[i], self.input_shape[0], self.input_shape[1]).to(self.device)
            )

    def setHidden(self, hidden):
        H, C = hidden
        self.H, self.C = H, C


class DownConv(nn.Module):
    def __init__(self, input_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                input_channels, 
                out_channels, 
                kernel_size=(3,3), 
                stride=stride, 
                padding=1
            ),
            nn.GroupNorm(16, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.conv(input)
    

class UpConv(nn.Module):
    def __init__(self, input_channels, out_channels, stride):
        super().__init__()
        if stride == 2:
            output_padding = 1
        else:
            output_padding = 0
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                input_channels, 
                out_channels, 
                kernel_size=(3,3), 
                stride=stride, 
                padding=1, 
                output_padding=output_padding
            ),
            nn.GroupNorm(16, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.conv(input)
        

class Encoder(nn.Module):
    def __init__(self, input_channels=1, phy_channels=32):
        super().__init__()
        # 下采样
        self.conv1 = DownConv(input_channels, phy_channels, stride=2)
        # 保持尺寸不变
        self.conv2 = DownConv(phy_channels, phy_channels, stride=1)
        # 下采样
        self.conv3 = DownConv(phy_channels, phy_channels*2, stride=2)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, input_channels=1, phy_channels=32):
        super().__init__()
        # 上采样
        self.conv1 = UpConv(phy_channels*2, phy_channels, stride=2)
        # 保持尺寸不变
        self.conv2 = UpConv(phy_channels, phy_channels, stride=1)
        # 上采样a
        self.conv3 = nn.ConvTranspose2d(
            in_channels=phy_channels,
            out_channels=input_channels,
            kernel_size=(3,3),
            stride=2,
            padding=1,
            output_padding=1
        )
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class PREncoder(nn.Module):
    def __init__(self, input_channels=64, phy_channels=64):
        super().__init__()
        self.conv1 = DownConv(input_channels, phy_channels, stride=1)
        self.conv2 = DownConv(phy_channels, phy_channels, stride=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        return x


class PRDecoder(nn.Module):
    def __init__(self, input_channels=64, phy_channels=64):
        super().__init__()
        self.conv1 = UpConv(phy_channels, phy_channels, stride=1)
        self.conv2 = UpConv(phy_channels, input_channels, stride=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        return x

def _apply_axis_left_dot(x, mats):
    assert x.dim() == len(mats)+1
    sizex = x.size()
    k = x.dim()-1
    for i in range(k):
        x = tensordot(mats[k-i-1], x, dim=[1,k])
    x = x.permute([k,]+list(range(k))).contiguous()
    x = x.view(sizex)
    return x


class _MK(nn.Module):
    def __init__(self, shape):
        super(_MK, self).__init__()
        self._size = torch.Size(shape)
        self._dim = len(shape)
        M = []
        invM = []
        assert len(shape) > 0
        j = 0
        for l in shape:
            # 创建转换矩阵M及其逆矩阵invM
            M.append(np.zeros((l,l)))
            for i in range(l):
                M[-1][i] = ((np.arange(l)-(l-1)//2)**i)/factorial(i)
            invM.append(np.linalg.inv(M[-1]))
            # 注册为buffer以便在GPU上使用
            self.register_buffer('_M'+str(j), torch.from_numpy(M[-1]))
            self.register_buffer('_invM'+str(j), torch.from_numpy(invM[-1]))
            j += 1

    @property
    def M(self):
        return list(self._buffers['_M'+str(j)] for j in range(self.dim()))

    @property
    def invM(self):
        return list(self._buffers['_invM'+str(j)] for j in range(self.dim()))

    def size(self):
        return self._size
    
    def dim(self):
        return self._dim
    
    def _packdim(self, x):
        assert x.dim() >= self.dim()
        if x.dim() == self.dim():
            x = x[np.newaxis,:]
        x = x.contiguous()
        x = x.view([-1,]+list(x.size()[-self.dim():]))
        return x

    def forward(self):
        pass


class M2K(_MK):
    def __init__(self, shape):
        super(M2K, self).__init__(shape)

    def forward(self, m):
        sizem = m.size()
        m = self._packdim(m)
        m = _apply_axis_left_dot(m, self.invM)
        m = m.view(sizem)
        return m


class K2M(_MK):
    def __init__(self, shape):
        super(K2M, self).__init__(shape)

    def forward(self, k):
        sizek = k.size()
        k = self._packdim(k)
        k = _apply_axis_left_dot(k, self.M)
        k = k.view(sizek)
        return k

def tensordot(a,b,dim):
    l = lambda x,y:x*y
    if isinstance(dim,int):
        # 处理整数维度参数的情况
        a = a.contiguous()
        b = b.contiguous()
        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-dim]
        sizea1 = sizea[-dim:]
        sizeb0 = sizeb[:dim]
        sizeb1 = sizeb[dim:]
        N = reduce(l, sizea1, 1)
        assert reduce(l, sizeb0, 1) == N  # 确保a和b的收缩维度匹配
    else:
        # 处理元组维度参数的情况
        adims = dim[0]
        bdims = dim[1]
        adims = [adims,] if isinstance(adims, int) else adims
        bdims = [bdims,] if isinstance(bdims, int) else bdims
        
        # 计算a的维度排列(非收缩维度在前，收缩维度在后)
        adims_ = set(range(a.dim())).difference(set(adims))
        adims_ = list(adims_)
        adims_.sort()
        perma = adims_+adims
        
        # 计算b的维度排列(收缩维度在前，非收缩维度在后)
        bdims_ = set(range(b.dim())).difference(set(bdims))
        bdims_ = list(bdims_)
        bdims_.sort()
        permb = bdims+bdims_
        
        # 重新排列维度
        a = a.permute(*perma).contiguous()
        b = b.permute(*permb).contiguous()

        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-len(adims)]
        sizea1 = sizea[-len(adims):]
        sizeb0 = sizeb[:len(bdims)]
        sizeb1 = sizeb[len(bdims):]
        N = reduce(l, sizea1, 1)
        assert reduce(l, sizeb0, 1) == N  # 确保a和b的收缩维度匹配
    
    # 执行矩阵乘法并恢复形状
    a = a.view([-1,N])
    b = b.view([N,-1])
    c = a@b
    return c.view(sizea0+sizeb1)