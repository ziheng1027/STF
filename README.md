# STF

一个新手向的时空预测框架: 时空预测, 视频序列预测, 雷达回波外推... 

## 数据集
### MovingMNIST:
- 下载(bash)：sh "Dataset\Download\download_moving_mnist.sh"
### TaxiBJ:
- 下载(bash)：sh "Dataset\Download\download_taxibj.sh"
### SEVIR:
- 运行download_sevir_vil.sh脚本之前, 请确保已安装AWS CLI: https://docs.aws.amazon.com/zh_cn/cli/latest/userguide/getting-started-install.html
- 下载(bash): sh "Dataset\Download\download_sevir_vil.sh"
- SEVIR数据处理可移步: https://github.com/ziheng1027/SEVIR/tree/main

## 模型
- ConvLSTM*(NIPS 2015)
- PredRNN(V1:NIPS 2017, V2:IEEE 2022)
- PhyDNet(CVPR 2020)
- UNet(CVPR 2015)
- SmaAtUNet(PRL 2021)
- SimVP(CVPR 2022)
- Tau*(CVPR 2023)
- STLight(WACV 2024)

## 训练器
- Trainer_Base: 训练器基类
- Trainer_DMF: 直接多步预测训练器, 适合seq2seq结构的模型
- Trainer_IMF: 迭代多步预测训练器, 适合自回归结构的模型

## 环境依赖
python>=3.12, cuda>=12.x, pytorch>=2.6.0  
1. pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128  
2. pip install -r requirements.txt  

## 运行说明
- Config目录包含模型配置文件(以数据集名称组织), 数据集配置文件以及指标配置文件(不同数据集可选择不同指标)
- 配置完成后, 在main脚本中指定model_name, dataset_name, mode=train开始训练
- 训练完成后, 打开Config中的模型配置文件, 将训练好的模型文件路径填入model_path, 然后在main脚本中选择mode=test开始测试
- test时可指定save_interval来规定样本保存间隔, test后可选择mode=visualize来可视化已保存的样本
- 输出统一存放在Output目录下, 包含训练完成的最佳模型文件, 训练日志, 训练损失曲线, 测试样本, 可视化等内容
- 参数说明: 
    - patience: 早停耐心值, 当超过多少个epoch没有提升时停止训练
    - resume_from: 断点续训, 意外中断训练时, 将当前最新的模型路径填入可以重新接着训练
    - model_path: 模型路径, 用于test模式测试指定模型的性能指标
    - save_interval: test模式下样本保存的间隔