import os
import torch
import numpy as np
from tqdm import tqdm
from Trainer.Trainer_Base import Trainer_Base
from Tool.Metric import cal_metrics
from Tool.Utils import plot_loss, save_test_samples


class Trainer_DMF(Trainer_Base):
    def __init__(self, config, datasets_config, dataset_name):
        super().__init__(config, datasets_config, dataset_name)

    def train_batch(self, data_batch):
        input, target = data_batch
        T_input, T_target = input.shape[1], target.shape[1]
        input, target = input.to(self.device), target.to(self.device)
        # 输入输出帧数相同的情况
        if T_input == T_target:
            output = self.model(input)
        # 输入帧数大于输出帧数的情况, 只取前T_target帧预测
        elif T_input > T_target:
            output = self.model(input)
            output = output[:, :T_target]

        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # OneCycleLR需要在每个batch后更新学习率
        self.update_scheduler(is_batch_update=True)
        
        return loss.item()
    
    def train_epoch(self, epoch):
        self.model.train()
        num_batch = len(self.train_loader)
        losses = []
        pbar = tqdm(self.train_loader, ncols=120)

        for idx, data_batch in enumerate(pbar):
            loss = self.train_batch(data_batch)
            losses.append(loss)
            pbar.set_description_str(f"Epoch[{epoch}/{self.config['epochs']}], Batch[{idx}/{num_batch}]")
            pbar.set_postfix_str(f"loss: {loss:.4f}, lr: {self.optimizer.param_groups[0]['lr']:.4f}")
        train_loss = np.mean(losses)

        # StepLR, ReduceLROnPlateau需要在每个epoch后更新学习率
        val_loss = self.validate()
        self.update_scheduler(val_loss, is_batch_update=False)

        return train_loss, val_loss

    def train(self):
        best_val_loss = float("inf")
        best_checkpoint_path = None

        if self.config["resume_from"] is not None:
            current_epoch = self.load_checkpoint(self.config["resume_from"])
            start_epoch = current_epoch + 1
        else:
            start_epoch = 1

        for epoch in range(start_epoch, self.config["epochs"] + 1):
            train_loss, val_loss = self.train_epoch(epoch)
            
            # 记录损失历史
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            
            self.logger.info(f"Epoch[{epoch}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, lr: {self.optimizer.param_groups[0]['lr']:.4f}")
            
            if self.early_stopping.check(val_loss):
                self.logger.info(f"Early stopping at epoch {epoch}!\n")
                break
            if val_loss < best_val_loss:
                if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
                    os.remove(best_checkpoint_path)
                best_val_loss = val_loss
                best_checkpoint_path = self.save_checkpoint(epoch, val_loss)
        
        # 训练结束后绘制损失曲线
        if self.train_loss_history and self.val_loss_history:
            plot_loss(self.train_loss_history, self.val_loss_history, self.model_name, self.dataset_name)
    
    def evaluate_batch(self, data_batch, mode="val"):
        input, target = data_batch
        T_input, T_target = input.shape[1], target.shape[1]
        input, target = input.to(self.device), target.to(self.device)
        output = self.model(input)
        # 输入输出帧数相同的情况
        if T_input == T_target:
            output = self.model(input)
        # 输入帧数大于输出帧数的情况, 只取前T_target帧预测
        elif T_input > T_target:
            output = self.model(input)
            output = output[:, :T_target]
            
        loss = self.criterion(output, target)

        if mode == "test":
            metrics_batch = cal_metrics(target.cpu().numpy(), output.cpu().numpy())
            return loss.item(), metrics_batch, input, target, output

        return loss.item()

    def validate(self):
        self.model.eval()
        losses = []
        pbar = tqdm(self.val_loader, desc="Val", ncols=100)

        with torch.no_grad():
            for data_batch in pbar:
                loss = self.evaluate_batch(data_batch, mode="val")
                losses.append(loss)
                pbar.set_postfix_str(f"loss: {loss:.4f}")

        return np.mean(losses)

    def test(self):
        self.load_model_weight(self.config["model_path"])
        self.model.eval()
        losses = []
        sample_idx = 0
        metrics = {"mse": [], "mae": [], "rmse": [], "psnr": [], "ssim": [], "lpips": []}
        pbar = tqdm(self.test_loader, desc="Test", ncols=100)

        with torch.no_grad():
            for data_batch in pbar:
                loss, metrics_batch, input, target, output = self.evaluate_batch(data_batch, mode="test")
                losses.append(loss)
                pbar.set_postfix_str(f"loss: {loss:.4f}")
                
                for key in metrics:
                    metrics[key].append(metrics_batch[key])
                
                sample_idx = save_test_samples(sample_idx, input, target, output, self.model_name, self.dirs['sample'], interval=64)
            
            avg_metrics = {}
            for key in metrics:
                avg_metrics[key] = round(np.mean(metrics[key]), 4)
            
            self.logger.log_metrics(avg_metrics)
            