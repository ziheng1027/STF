# STF

时空预测: 初学者入门学习记录(简单框架)

## 数据集
如果下载速度过慢,建议使用google colab下载
### MovingMNIST:
- 下载：Data\download_moving_mnist.sh
### TaxiBJ:
- 下载：Data\download_taxibj.sh

## 模型
- UNet
- SmaAt-UNet
- PhyDNet
- MLPMixer*
- ConvMixer*
- ConvNeXt*
- SimVP
- Tau*
- STLight*

## 环境配置
python-3.13.5: conda install
cuda-12.8: conda install
pytorch-2.8.0: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
numpy-2.1.2: conda install
pyyaml-6.0.2: conda install