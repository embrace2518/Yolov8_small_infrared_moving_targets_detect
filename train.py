import os
import torch
from ultralytics import YOLO


last_weights = "runs/detect/train/weights/last.pt"
base_model = "models/yolov8.yaml"
pretrained_weights = "models/yolov8n.pt"

train_device = 0 if torch.cuda.is_available() else "cpu"
train_args = dict(
    data="dataset/infrared.yaml",
    epochs=100,
    imgsz=640,
    optimizer="AdamW",
    batch=8,
    lr0=0.001,
    warmup_epochs=3,
    close_mosaic=10,
    mosaic=0.3,
    mixup=0.05,
    fliplr=0,
    erasing=0.1,
    cos_lr=True,
    augment=True,
    multi_scale=False,
    half=torch.cuda.is_available(),
    device=train_device,
    workers=0,
)
def is_checkpoint_valid(checkpoint_path):
    try:
        torch.load(checkpoint_path, map_location="cpu")
        return True
    except Exception:
        return False

def load_model():
    if os.path.exists(last_weights) and is_checkpoint_valid(last_weights):
        print(f"[train] Resume candidate found: {last_weights}")
        model = YOLO(last_weights)
    else:
        print("[train] No valid checkpoint found. Starting fresh training.")
        model = YOLO(base_model).load(pretrained_weights)  # build from YAML and transfer weights
    model.info()
    return model

def train_model(model):
    try:
        results = model.train(resume=True, **train_args)
    except AssertionError as e:
        # Ultralytics may reject resume when the previous run already reached target epochs.
        if "nothing to resume" in str(e).lower():
            print("[train] Previous run already finished. Falling back to fresh training.")
            model = YOLO(base_model).load(pretrained_weights)
            results = model.train(resume=False, **train_args)
        else:
            raise

