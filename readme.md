

```
python my_anno.json
python validation.py my_anno.json true_labels.json
```

## 红外数据预处理

新增文件：`dataset/preprocess_ir.py`

处理流程：
- 非均匀校正（场景自适应 NUC）
- 噪声抑制（Gaussian / Median / NLM）
- 动态范围压缩与对比度增强（CLAHE + Gamma）
- 背景抑制（GMM / MOG2）
- 目标初筛（连通域候选框）

默认配置文件：`dataset/preprocess_config.yaml`

### 自检运行

```powershell
python dataset\preprocess_ir.py --self-test
```

### 处理真实数据

```powershell
python dataset\preprocess_ir.py --config dataset\preprocess_config.yaml --input D:\Dataset\train --output D:\Dataset_preprocessed\train
```

输出目录结构：
- `enhanced/序列号/`：增强后的训练图像

## proposals/labels 生成（统一脚本）

使用文件：`dataset/synthetic_augment.py`

作用：从 `enhanced/<序列>/<帧>` 生成 `masks/overlays/proposals`，并可选直接转换为 YOLO 标签 `labels/<序列>/<帧>.txt`。

### 自检运行

```powershell
python dataset\synthetic_augment.py --self-test --convert-labels
```

### 训练集：生成 proposals 并转换 labels（推荐）

```powershell
python dataset\synthetic_augment.py --config dataset\preprocess_config.yaml --input D:\Dataset_preprocessed\train\enhanced --output D:\Dataset_preprocessed\train --convert-labels --score-thresh 5.0 --class-id 0 --max-objects 20
```

### 仅将已有 proposals 转为 labels（可选）

```powershell
python dataset\synthetic_augment.py --input D:\Dataset_preprocessed\train\enhanced --output D:\Dataset_preprocessed\train --convert-labels --proposal-root D:\Dataset_preprocessed\train\proposals --label-root D:\Dataset_preprocessed\train\labels --score-thresh 5.0 --class-id 0 --max-objects 20
```

