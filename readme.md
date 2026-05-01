# Yolov8 Small Infrared Moving Targets Detect

## 1) 数据准备

坐标都归一化到 `[0,1]`。

## 2) 预处理

处理流程：
- NUC（非均匀校正）
- 去噪（Gaussian / Median / NLM）
- CLAHE + Gamma 增强

## 3) 训练

```powershell
python train.py --config train_config.yaml

python train.py --config train_config.yaml --resume runs/detect/exp_20260501_173906/weights/best.pt
```

## 4) 评估

```powershell
python evaluate.py --weights runs/detect/exp_20260429_094305/weights/best.pt --config train_config.yaml


