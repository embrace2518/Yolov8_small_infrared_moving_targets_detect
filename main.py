
from ultralytics import YOLO
from dataset.convert_json import model_validate
from train import load_model

if __name__ == '__main__':
    model = load_model()

    # val_results = model.val()

    predictions = model.predict(source=r"D:\Dataset\train\10", save=True, conf=0.1)
    model_validate(predictions)

    # success = model.export(format="onnx")
