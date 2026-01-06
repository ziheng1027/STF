# STF

时空预测: 学习记录

## 数据集
### MovingMNIST:
- 下载(bash)：sh "Dataset\Download\download_moving_mnist.sh"
### TaxiBJ:
- 下载(bash)：sh "Dataset\Download\download_taxibj.sh"
### SEVIR:
- 运行download_sevir_vil.sh脚本之前, 请确保已安装AWS CLI: https://docs.aws.amazon.com/zh_cn/cli/latest/userguide/getting-started-install.html
- 下载(bash): sh "Dataset\Download\download_sevir_vil.sh"

## 模型
- ConvLSTM*
- PredRNN*
- PhyDNet
- UNet
- SmaAtUNet
- SimVP
- Tau*
- STLight*

## 环境依赖
python>=3.12, cuda>=12.x, pytorch>=2.6.0  
1. pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128  
2. pip install -r requirements.txt  

## 运行说明
- Config目录包含模型配置文件(以数据集名称组织), 数据集配置文件以及指标配置文件(不同数据集可选择不同指标)
- 配置完成后, 在main脚本中选择模型, 数据集, 训练/测试开始运行
- 输出统一存放在Output目录下, 包含训练完成的最佳模型文件, 训练日志, 训练曲线等内容