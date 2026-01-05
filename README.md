# STF

时空预测: 学习记录

## 数据集
### MovingMNIST:
- 下载(bash)：sh Data\download_moving_mnist.sh
### TaxiBJ:
- 下载(bash)：sh Data\download_taxibj.sh
### SEVIR:
- 运行download_sevir_vil.sh脚本之前, 请确保已安装AWS CLI: https://docs.aws.amazon.com/zh_cn/cli/latest/userguide/getting-started-install.html
- 下载(bash): sh Data\download_sevir_vil.sh

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
(cuda>=12.x, pytorch>=2.6.0): pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 
pip install -r requirements.txt