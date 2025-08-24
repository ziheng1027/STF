# STF

时空预测: 初学者入门学习记录(简单框架)

## 数据集
如果下载速度过慢,建议在google colab中下载
### MovingMNIST:
- 下载：Data\download_moving_mnist.sh
### TaxiBJ:
- 下载：Data\download_taxibj.sh

## 模型
- ConvLSTM*
- PredRNN*
- UNet
- SmaAt-UNet
- PhyDNet
- MLPMixer*
- ConvMixer*
- ConvNeXt*
- SimVP
- Tau*
- STLight*

## 环境依赖
python-3.13.5  
(cuda-12.8, pytorch-2.8.0): pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128  
numpy-2.1.2  
matplotlib-3.10.0  
scipy-1.16.1  
scikit-image-0.25.2
pyyaml-6.0.2  
tqdm-4.67.1  
lpips-0.1.4