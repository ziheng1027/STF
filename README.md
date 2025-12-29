# STF

时空预测: 初学者入门学习记录(简单框架)

## 数据集
### MovingMNIST:
- 下载：Data\download_moving_mnist.sh
### TaxiBJ:
- 下载：Data\download_taxibj.sh
### SEVIR:
- 首先确保已安装AWS CLI: https://docs.aws.amazon.com/zh_cn/cli/latest/userguide/getting-started-install.html
- 下载: Data\download_sevir_vil.sh

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
python>=3.12
(cuda=12.6, pytorch=2.6.0): pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 
