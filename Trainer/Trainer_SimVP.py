# Trainer/Trainer_SimVP.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from Trainer.Trainer_DMF import Trainer_DMF


class DDRLoss(nn.Module):
    """计算帧间差分散度正则化损失, TAU模型需要的额外损失"""
    def __init__(self, alpha=0.1, tau=0.1, eps=1e-12):
        super().__init__()

        self.mse = nn.MSELoss()
        # 正则化项的权重系数
        self.alpha = alpha
        # 温度系数τ, 用于控制Softmax分布的锐度, 较小的τ会放大差异使模型关注剧烈的运动变化
        self.tau = tau
        # 避免除零错误
        self.eps = eps
    
    def diff_div_reg(self, output, target, ):
        """计算差分散度正则化"""
        B, T, C = output.shape[:3]
        # 如果序列长度小于2, 则无法计算差分, 返回0
        if T <= 2:
            return 0
        
        # 计算帧间差分
        gap_output = (output[:, 1:] - output[:, :-1]).reshape(B, T-1, -1) # (B, T-1, C*H*W)
        gap_target = (target[:, 1:] - target[:, :-1]).reshape(B, T-1, -1)

        #将差分转换为概率分布
        softmax_gap_output = F.softmax(gap_output / self.tau, dim=-1)
        softmax_gap_target = F.softmax(gap_target / self.tau, dim=-1)

        # 计算KL散度
        ddr = softmax_gap_output * torch.log(softmax_gap_output / (softmax_gap_target + self.eps) + self.eps)
        return ddr.mean()

    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        ddr_loss = self.diff_div_reg(output, target)
        total_loss = mse_loss + self.alpha * ddr_loss
        return total_loss


class Trainer(Trainer_DMF):
    def __init__(self, model_config, datasets_config, metric_config, dataset_name):
        super().__init__(model_config, datasets_config, metric_config, dataset_name)

    def get_model(self):
        from Model.SimVP import SimVP
        model = SimVP(**self.model_config['model'])
        return model.to(self.device), model.__class__.__name__
    
    def get_criterion(self):
        translator_type = self.model_config['model']['translator_type']
        if translator_type.lower() == "tau":
            alpha = self.model_config.get('alpha', 0.1)
            tau = self.model_config.get('tau', 0.1)
            return DDRLoss(alpha=alpha, tau=tau).to(self.device)
        
        return super().get_criterion()