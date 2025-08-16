import os
import torch
import logging
import torch.nn as nn
import torch.optim as optim

class Trainer_Base:
    """训练器基类"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.model_name = self.get_model()
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloader(config['dataset'])
        self.optimizer = self.get_optimizer(config['train']['optimizer'])
        self.scheduler = self.get_scheduler(config['train']['scheduler'])
        self.criterion = self.get_criterion()

    def get_model(self):
        """获取模型"""
        pass

    def get_dataloader(self, dataset_name):
        """获取数据加载器"""
        pass

    def get_optimizer(self, optimizer_name):
        """获取优化器"""
        pass

    def get_scheduler(self, scheduler_name):
        """获取学习率调度器"""
        pass

    def get_criterion(self):
        """获取损失函数"""
        return nn.MSELoss()

    def save_checkpoint(self, epoch, val_loss):
        """保存模型检查点"""
        pass

    def load_checkpoint(self, epoch):
        """加载模型检查点, 断点续训"""
        pass

    def train_batch(self, batch):
        """训练一个batch"""
        pass

    def train_epoch(self, epoch):
        """训练一个epoch"""
        pass

    def train(self):
        """训练模型"""
        pass

    def evaluate_batch(self, batch):
        """评估一个batch"""
        pass

    def evaluate_epoch(self, epoch):
        """评估一个epoch"""
        pass

    def validate(self):
        """验证模型"""
        pass

    def test(self):
        """测试模型"""
        pass