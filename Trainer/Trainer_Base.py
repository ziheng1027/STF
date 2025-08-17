import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Tool.Logger import Logger

class Trainer_Base:
    """训练器基类"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dirs = self.make_dirs()
        self.model, self.model_name = self.get_model()
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloader(config['dataset'])
        self.optimizer = self.get_optimizer(config['train']['optimizer'])
        self.scheduler = self.get_scheduler(config['train']['scheduler'])
        self.criterion = self.get_criterion()
        self.logger = Logger(self.model_name)

    def get_model(self):
        """获取模型"""
        pass

    def get_dataloader(self, dataset_name):
        """获取数据加载器"""
        pass

    def get_optimizer(self, optimizer_name):
        """获取优化器"""
        if optimizer_name == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['train'].get("lr", 0.002))
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(self.model.parameters(), lr=self.config['train'].get("lr", 0.002))
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.config['train'].get("lr", 0.002), momentum=0.9)
        else:
            raise ValueError(f"不支持的optimizer: {optimizer_name}")
        
        return optimizer
    
    def get_scheduler(self, scheduler_name):
        """获取学习率调度器"""
        if scheduler_name == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['train'].get("step_size", 10))
        elif scheduler_name == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        elif scheduler_name == "OneCycleLR":
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['train'].get("max_lr", 0.1),
                epochs=self.config['train'].get("epochs", 100),
                steps_per_epoch=len(self.train_loader)
            )
        else:
            raise ValueError(f"不支持的scheduler: {scheduler_name}")
        
        return scheduler

    def get_criterion(self):
        """获取损失函数"""
        return nn.MSELoss()

    def make_dirs(self):
        """集中创建所需目录"""
        dirs = {
            "root": "./Output",
            "checkpoint": "./Output/Checkpoint",
            "log": "./Output/Log",
            "visualization": "./Output/Visualization",
            "result": "./Output/Result"
        }
        for dir in dirs.values():
            os.makedirs(dir, exist_ok=True)
        return dirs

    def save_checkpoint(self, epoch, val_loss):
        """保存模型检查点"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "criterion_state_dict": self.criterion.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "random_python": random.getstate(),
            "random_numpy": np.random.get_state(),
            "random_torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        checkpoint_path = f"{self.dirs['checkpoint']}/{self.model_name}_epoch{epoch}_loss{val_loss}.pt"
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点, 断点续训"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"未找到检查点文件: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.criterion.load_state_dict(checkpoint["criterion_state_dict"])
        current_epoch = checkpoint["epoch"]

        random.setstate(checkpoint["random_python"])
        np.random.set_state(checkpoint["random_numpy"])
        torch.set_rng_state(checkpoint["random_torch"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda"])

        self.logger.info(f"加载检查点成功, epoch: {self.current_epoch}, val_loss: {self.val_loss}, path: {checkpoint_path}")

        return current_epoch

    def load_model_weight(self, checkpoint_path):
        """加载模型权重, 测试模型"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"未找到检查点文件: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.logger.info(f"加载模型权重成功, val_loss: {self.val_loss}, path: {checkpoint_path}")

    def train_batch(self, data_batch):
        """训练一个batch"""
        pass

    def train_epoch(self, epoch):
        """训练一个epoch"""
        pass

    def train(self):
        """训练模型"""
        pass

    def evaluate_batch(self, data_batch, mode="val"):
        """评估一个batch"""
        pass

    def evaluate_epoch(self, epoch, mode="val"):
        """评估一个epoch"""
        pass

    def validate(self):
        """验证模型"""
        pass

    def test(self):
        """测试模型"""
        pass