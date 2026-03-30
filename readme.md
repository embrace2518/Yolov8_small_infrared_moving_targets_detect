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

## 2) 预处理（可选）

脚本：`dataset/preprocess.py`

处理流程：
- NUC（非均匀校正）
- 去噪（Gaussian / Median / NLM）
- CLAHE + Gamma 增强

运行示例：

```powershell
python dataset\preprocess.py --config dataset\preprocess_config.yaml --input D:\Dataset\train --output D:\Dataset_preprocessed\train
```

## 3) 训练

入口：`train.py`

说明：训练前会用 `EnhancedYOLODataset + DataLoader` 做预处理并暂存到 `runs/detect/staging/...`，然后调用 Ultralytics `YOLO.train()` 训练。

```powershell
python train.py --config train_config.yaml
python train.py --config train_config.yaml --debug-cuda --safe
python train.py --config train_config.yaml --device cpu
python train.py --config train_config.yaml --resume runs\detect\exp_xxx\weights\last.pt
```

## 4) 推理

入口：`main.py`

```powershell
python main.py --model models\yolov8.pt --source D:\Dataset\test\1 --save --conf 0.1
```

预测图默认保存在 `runs/detect/predict*`。
