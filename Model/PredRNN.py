# Model/PredRNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from Module.PredRNN import STLSTMCell


class PredRNN(nn.Module):
    def __init__(self, input_channels, num_hidden_channels, input_frames, output_frames, patch_size, img_height, img_width, 
                 kernel_size, stride, use_layer_norm=True, reverse_scheduled_sampling=True, model_version="V1"):
        super().__init__()

        self.num_hidden_channels = num_hidden_channels
        self.layers = len(num_hidden_channels)
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.total_frames = input_frames + output_frames
        self.patch_size = patch_size
        self.height = img_height // patch_size
        self.width = img_width // patch_size
        self.patched_channels = input_channels * patch_size * patch_size
        self.reverse_scheduled_sampling = reverse_scheduled_sampling
        self.model_version = model_version

        cell_list = []
        for i in range(self.layers):
            in_channels = self.patched_channels if i == 0 else num_hidden_channels[i - 1]
            out_channels = num_hidden_channels[i]

            cell_list.append(
                STLSTMCell(
                    in_channels=in_channels,
                    hidden_channels=out_channels,
                    height=self.height,
                    width=self.width,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    use_layer_norm=use_layer_norm,
                    version=model_version
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

        if self.model_version == "V2":
            # Adapter用于将delta_c 和 delta_m 映射到同一空间以便计算decouple_loss
            self.adapter = nn.Conv2d(
                in_channels=num_hidden_channels[0],
                out_channels=num_hidden_channels[0],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )

        # 输出卷积
        self.conv_out = nn.Conv2d(
            in_channels=num_hidden_channels[-1],
            out_channels=self.patched_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
    
    def forward(self, x_patched, mask_patched=None):
        # 初始化状态
        B, T, _, H, W = x_patched.shape
        x_next_patched = []

        h_t = [torch.zeros(B, self.num_hidden_channels[i], H, W, device=x_patched.device) for i in range(self.layers)]
        c_t = [torch.zeros(B, self.num_hidden_channels[i], H, W, device=x_patched.device) for i in range(self.layers)]
        memory = torch.zeros(B, self.num_hidden_channels[0], H, W, device=x_patched.device)

        # PredRNN-V2计算decouple_loss
        if self.model_version == "V2":
            decouple_loss_list = [] # length = (total_frames - 1) * layers
            
        # 时间步迭代
        for t in range(self.total_frames - 1):
            # reverse mode(从 预测值 -> 真值, 具体概率在外部实现): 在前input_frames输入阶段也采样预测结果作为下一步输入, 在观察阶段就接触预测出的“烂图”，从而学出更鲁棒的特征
            if self.reverse_scheduled_sampling:
                if t == 0:  # 第一帧总是用真实值
                    x_t = x_patched[:, t]
                else:
                    if mask_patched is not None:  # mask_patched是一个和x_patched相同形状的二值张量(全0/1)
                        # 这里的mask取t - 1是因为mask_patched的长度是total_frames - 1(模型输入1-19帧, 预测2-20帧)
                        mask_t = mask_patched[:, t - 1]
                        x_t = mask_t * x_patched[:, t] + (1 - mask_t) * x_out
                    else:
                        x_t = x_out
            # standard mode(从 真值 -> 预测值): 在前input_frames输入阶段只能使用真实值作为下一步输入, 在预测阶段才开始采样预测结果作为下一步输入
            else:
                if t < self.input_frames:   # 前input_frames帧用真实值
                    x_t = x_patched[:, t]
                else:
                    if mask_patched is not None:
                        # 此处mask取t - input_frames和t - 1并没有区别, 因为mask是随机生成的全1或全0矢量, 但取t - input_frames可以兼容只传入input_frames帧mask的情况
                        mask_t = mask_patched[:, t - self.input_frames]
                        x_t = mask_t * x_patched[:, t] + (1 - mask_t) * x_out
                    else:
                        x_t = x_out
            
            # layer 0, PredRNN-V2 需要获取 delta_c 和 delta_m 来计算decouple_loss
            if self.model_version == "V2":
                h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](x_t, h_t[0], c_t[0], memory)
            else:
                h_t[0], c_t[0], memory = self.cell_list[0](x_t, h_t[0], c_t[0], memory)
            
            if self.model_version == "V2":
                delta_c_norm = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_norm = F.normalize(self.adapter(delta_m).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                decouple_loss_list.append(torch.mean(torch.abs(torch.cosine_similarity(delta_c_norm, delta_m_norm, dim=2))))
            
            # layer 1~N
            for i in range(1, self.layers):
                if self.model_version == "V2":
                    h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                    delta_c_norm = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                    delta_m_norm = F.normalize(self.adapter(delta_m).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                    decouple_loss_list.append(torch.mean(torch.abs(torch.cosine_similarity(delta_c_norm, delta_m_norm, dim=2))))
                else:
                    h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_out = self.conv_out(h_t[-1])
            x_next_patched.append(x_out)
        
        x_next_patched = torch.stack(x_next_patched, dim=1)  # [B, T-1, C_patched, H, W]
        # 如果是PredRNN-V2, 计算平均解耦损失
        decouple_loss = 0
        if self.model_version == "V2":
            decouple_loss = torch.mean(torch.stack(decouple_loss_list))
            print("Decouple Loss List:", decouple_loss_list)

        return x_next_patched, decouple_loss

if __name__ == "__main__":
    import sys
    import os
    from Tool.Utils import patchify, unpatchify
    
    # 确保可以导入 Module
    sys.path.append(os.getcwd())
    from torchinfo import summary

    # 1. 定义超参数 (模拟 Moving MNIST 实验设置)
    B, C, T, H, W = 4, 1, 20, 64, 64
    input_frames = 10
    output_frames = 10
    patch_size = 4
    num_hidden_channels = [64, 64, 64, 64] # 4层网络

    # 2. 实例化模型
    # 注意这里使用 hidden_channels 参数名，匹配你上传的代码
    model = PredRNN(
        input_channels=C,
        num_hidden_channels=num_hidden_channels,
        input_frames=input_frames,
        output_frames=output_frames,
        patch_size=patch_size,
        img_height=H,
        img_width=W,
        kernel_size=5,
        stride=1,
        use_layer_norm=True,
        reverse_scheduled_sampling=True,
        model_version="V2"  # 可以切换 "V1" 或 "V2"
    ).cuda()

    # 3. 打印模型结构
    # 输入形状: [Batch, Total_Frames, Channels, Height, Width]
    input_shape = (B, T, C * (patch_size ** 2), H // patch_size, W // patch_size)
    
    print("\n" + "="*50)
    print("PredRNN Model Summary")
    print("="*50)
    summary(model, input_size=input_shape)

    # 4. 简单的前向传播测试
    dummy_input = torch.randn(B, T, C, H, W).cuda()
    input_patched = patchify(dummy_input, patch_size)  # [B, T, C*(p*p), H/p, W/p]
    with torch.no_grad():
        output_patched, decouple_loss = model(input_patched)
    output = unpatchify(output_patched, patch_size)  # [B, T-1, C, H, W]

    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Input_patched Shape:  {input_patched.shape}")
    print(f"Output_patched Shape: {output_patched.shape}")
    print(f"Output Shape: {output.shape}")