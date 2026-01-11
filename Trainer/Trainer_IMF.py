# Trainer/Trainer_IMF.py
import os
import torch
import numpy as np
from tqdm import tqdm
from Trainer.Trainer_Base import Trainer_Base
from Tool.Metric import cal_metrics
from Tool.Utils import plot_loss, save_test_samples, patchify, unpatchify, get_sampling_threshold


class Trainer_IMF(Trainer_Base):
    def __init__(self, model_config, dataset_config, metric_config, dataset_name):
        super().__init__(model_config, dataset_config, metric_config, dataset_name)

        self.steps = 0  # 用于记录训练步数, 以便调整采样率

    def train_batch(self, data_batch):
        input, target = data_batch
        input, target = input.to(self.device), target.to(self.device)
        combine = torch.cat([input, target], dim=1)  # [B, T_input + T_target, C, H, W]
        input_patched = patchify(combine, self.model_config["model"]["patch_size"])  # [B, T, C_patched, H_patched, W_patched]

        # scheduled sampling 掩码生成
        B, T, C, H, W = input_patched.shape
        sampling_threshold = get_sampling_threshold(
            self.steps, 
            self.model_config["start_iter"],
            self.model_config["end_iter"],
            self.model_config["model"].get("reverse_scheduled_sampling", True)
        )
        mask_patched = torch.zeros((B, T - 1, C, H, W)).to(self.device)
        random_flip = torch.rand((B, T - 1)).to(self.device)
        ones = torch.ones_like(random_flip).to(self.device)
        zeros = torch.zeros_like(random_flip).to(self.device)
        mask_flag = torch.where(random_flip < sampling_threshold, ones, zeros)
        mask_patched = mask_flag.view(B, T - 1, 1, 1, 1).expand_as(mask_patched)

        # 正常scheduled sampling(前input_frames-1帧)强制使用真值
        if not self.model_config["model"]["reverse_scheduled_sampling"]:
            mask_patched[:, :self.model_config["model"]["input_frames"] - 1] = 1.0
        
        output_patched, aux_loss = self.model(input_patched, mask_patched)
        loss = self.criterion(output_patched, input_patched[:, 1:])  # 预测输出(2->20, 19帧)
        total_loss = loss + self.model_config["aux_loss_weight"] * aux_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        # OneCycleLR需要在每个batch后更新学习率
        self.update_scheduler(is_batch_update=True)
        self.steps += 1

        return total_loss.item()


    def train_epoch(self, epoch):
        self.model.train()
        num_batch = len(self.train_loader)
        losses = [] # 记录每个batch的训练集损失
        pbar = tqdm(self.train_loader, ncols=150)

        for idx, data_batch in enumerate(pbar):
            loss = self.train_batch(data_batch)
            losses.append(loss)
            pbar.set_description_str(f"Epoch[{epoch}/{self.model_config['epochs']}], Batch[{idx}/{num_batch}]")
            pbar.set_postfix_str(f"loss: {loss:.4f}, lr: {self.optimizer.param_groups[0]['lr']:.4f}")
        train_loss = np.mean(losses)
        valid_loss = self.validate()
        # StepLR, ReduceLROnPlateau需要在每个epoch后更新学习率
        self.update_scheduler(valid_loss, is_batch_update=False)

        return train_loss, valid_loss

    def evaluate_batch(self, data_batch, mode="val"):
        input, target = data_batch
        input, target = input.to(self.device), target.to(self.device)
        combine = torch.cat([input, target], dim=1)  # [B, T_input + T_target, C, H, W]
        input_patched = patchify(combine, self.model_config["model"]["patch_size"])  # [B, T, C_patched, H_patched, W_patched]

        # 验证阶段前input_frames-1帧全部使用真值, 后续步骤全部使用预测值
        B, T, C, H, W = input_patched.shape
        mask_patched = torch.zeros((B, T - 1, C, H, W)).to(self.device)
        mask_patched[:, :self.model_config["model"]["input_frames"] - 1] = 1.0

        with torch.no_grad():
            output_patched, _ = self.model(input_patched, mask_patched)
        # 只取预测的后T_target帧
        output_patched = output_patched[:, self.model_config["model"]["input_frames"] - 1:]
        target_patched = input_patched[:, self.model_config["model"]["input_frames"]:]
        loss = self.criterion(output_patched, target_patched)

        # 将模型输出的切块恢复到原始尺寸
        output = unpatchify(output_patched, self.model_config["model"]["patch_size"])
        
        # 测试时额外返回每个batch的指标以及输入, 目标和输出用于保存样本用于可视化
        if mode == "test":
            metrics_batch = cal_metrics(target.cpu().numpy(), output.cpu().numpy(), self.metrics_name, self.threshold)
            return loss.item(), metrics_batch, input, target, output
        
        return loss.item()

    def train(self):
        best_valid_loss = float("inf")
        best_checkpoint_path = None

        # 判断是否进行断点续训
        if self.model_config["resume_from"] is not None:
            current_epoch = self.load_checkpoint(self.model_config["resume_from"])
            start_epoch = current_epoch + 1
        else:
            start_epoch = 1
        
        # 模型训练-主循环
        for epoch in range(start_epoch, self.model_config["epochs"] + 1):
            train_loss, valid_loss = self.train_epoch(epoch)
            
            # 记录每个epoch的损失
            self.train_loss_history.append(train_loss)
            self.valid_loss_history.append(valid_loss)
            
            self.logger.info(f"Epoch[{epoch}], Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, lr: {self.optimizer.param_groups[0]['lr']:.4f}")
            
            # 判断是否早停
            if self.early_stopping.check(valid_loss):
                self.logger.info(f"Early stopping at epoch {epoch}!\n")
                break

            # 保存最佳模型并删除旧的最佳模型
            if valid_loss < best_valid_loss:
                if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
                    os.remove(best_checkpoint_path)
                best_valid_loss = valid_loss
                best_checkpoint_path = self.save_checkpoint(epoch, valid_loss)
        
        # 训练结束后绘制损失曲线
        if self.train_loss_history and self.valid_loss_history:
            plot_loss(self.train_loss_history, self.valid_loss_history, self.model_name, self.dataset_name)

    def validate(self):
        self.model.eval()
        losses = [] # 记录每个batch的验证集损失
        pbar = tqdm(self.valid_loader, desc="Valid", ncols=150)

        with torch.no_grad():
            for data_batch in pbar:
                loss = self.evaluate_batch(data_batch, mode="val")
                losses.append(loss)
                pbar.set_postfix_str(f"loss: {loss:.4f}")
            
        return np.mean(losses)

    def test(self):
        self.load_model_weight(self.model_config["model_path"])
        self.model.eval()
        losses = [] # 记录每个batch的测试集损失
        sample_idx = 0
        metrics = {name: [] for name in self.metrics_name}
        pbar = tqdm(self.test_loader, desc="Test", ncols=150)

        with torch.no_grad():
            for data_batch in pbar:
                loss, metrics_batch, input, target, output = self.evaluate_batch(data_batch, mode="test")
                losses.append(loss)
                pbar.set_postfix_str(f"loss: {loss:.4f}")
                
                for key in metrics:
                    if key in metrics_batch:
                        metrics[key].append(metrics_batch[key])
                
                sample_idx = save_test_samples(sample_idx, input, target, output, self.model_name, self.dirs['sample'], interval=64)
            
            avg_metrics = {}
            for key in metrics:
                avg_metrics[key] = round(np.mean(metrics[key]), 4)
            
            self.logger.log_metrics(avg_metrics)