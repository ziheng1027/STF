import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Tool.Logger import Logger


class Trainer_Base:
    """训练器基类"""
    def __init__(self, config, datasets_config, dataset_name):
        self.config = config    # 训练配置
        self.datasets_config = datasets_config  # 所有数据集的配置
        self.dataset_name = dataset_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.model_name = self.get_model()
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloader(dataset_name)
        self.optimizer = self.get_optimizer(config['train']['optimizer'])
        self.scheduler = self.get_scheduler(config['train']['scheduler'])
        self.criterion = self.get_criterion()
        
        self.dirs = self.make_dirs()
        self.logger = Logger(self.model_name, self.dirs["log"])

    def get_model(self):
        """获取模型"""
        pass

    def get_dataloader(self, dataset_name):
        """获取数据加载器"""
        dataset_config = self.datasets_config[dataset_name]
        if dataset_name == "MovingMNIST":
            from Dataset.MovingMNIST import get_dataloader
            return get_dataloader(**dataset_config)
        elif dataset_name == "TaxiBJ":
            from Dataset.TaxiBJ import get_dataloader
            return get_dataloader(**dataset_config)
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")

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
    
    def update_scheduler(self, val_loss=None, is_batch_update=False):
        """更新学习率调度器"""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
            if not is_batch_update:
                self.scheduler.step()
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            if is_batch_update:
                self.scheduler.step()
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if not is_batch_update:
                if val_loss is None:
                    raise ValueError("请提供val_loss以更新ReduceLROnPlateau")
                self.scheduler.step(val_loss)

    def get_criterion(self):
        """获取损失函数"""
        return nn.MSELoss()

    def make_dirs(self):
        """集中创建所需目录"""
        dirs = {
            "root": "./Output",
            "checkpoint": f"./Output/Checkpoint/{self.dataset_name}",
            "log": f"./Output/Log/{self.dataset_name}",
            "visualization": f"./Output/Visualization/{self.dataset_name}/{self.model_name}",
            "sample": f"./Output/Sample/{self.dataset_name}/{self.model_name}"
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
        checkpoint_path = f"{self.dirs['checkpoint']}/{self.model_name}_epoch{epoch}_loss{val_loss:.5f}.pt"
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点, 断点续训"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"未找到检查点文件: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

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

        self.logger.info(f"Checkpoint loaded successfully! epoch: {self.current_epoch}, val_loss: {self.val_loss}, path: {checkpoint_path}")

        return current_epoch

    def load_model_weight(self, checkpoint_path):
        """加载模型权重, 测试模型"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"未找到检查点文件: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.logger.info(f"Model weights loaded successfully! path: {checkpoint_path}")

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

    def validate(self):
        """验证模型"""
        pass

    def test(self):
        """测试模型"""
        pass
