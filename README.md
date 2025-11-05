# CUDA 训练示例

## 依赖
- Python 3.9+
- PyTorch (需安装支持CUDA的版本)
- torchvision

可以使用如下命令安装基础依赖：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> 请根据自己的CUDA版本调整以上命令中的`cu124`后缀。

## 运行
使用`train_cuda.py`脚本在GPU上训练MNIST分类模型：

```bash
python train_cuda.py --epochs 10 --batch-size 256 --use-amp
```

常用参数：
- `--data-dir`: MNIST数据下载与缓存目录，默认`./data`
- `--output-dir`: 模型和日志保存目录，默认`./outputs`
- `--epochs`: 训练轮数，默认5
- `--batch-size`: 批大小，默认128
- `--lr`: 学习率，默认1e-3
- `--use-amp`: 开启混合精度训练（推荐在支持Tensor Core的GPU上使用）

脚本会在验证集上达到更高准确率时自动保存最优模型至`best_model.pt`。
