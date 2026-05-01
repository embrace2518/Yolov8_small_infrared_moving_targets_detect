# Yolov8 Small Infrared Moving Targets Detect

## 1) 数据准备

训练/验证目录使用 YOLO 标准结构：

```text
<root>/
  images/
  labels/
```

其中 `labels/*.txt` 为 YOLO 格式：

```text
class_id x_center y_center width height
```

坐标都归一化到 `[0,1]`。

## 2) 预处理

处理流程：
- NUC（非均匀校正）
- 去噪（Gaussian / Median / NLM）
- CLAHE + Gamma 增强

## 3) 训练

入口：`train.py`

```powershell
python train.py --config train_config.yaml

python train.py --config train_config.yaml --resume runs/detect/exp_20260429_094305/weights/last.pt
```

## 4) 评估


```powershell
python evaluate.py --weights runs/detect/exp_20260429_094305/weights/best.pt --config train_config.yaml

# 自定义参数
python evaluate.py --weights runs/detect/exp/weights/best.pt ^
    --data D:/Dataset/val ^
    --preprocess ^
    --batch-size 64 ^
    --output-dir my_eval ^
    --max-visualize 50
```


