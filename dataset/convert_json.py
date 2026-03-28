import json
from pathlib import Path

def model_validate(predictions):
    validation_data = []
    for result in predictions:
        img_path = result.path
        sequence_id = int(Path(img_path).parent.name)
        frame = int(Path(img_path).stem)
        boxes = result.boxes
        object_coords = []
        if boxes is not None:
            for box in boxes.xyxy:  # xyxy 格式
                x1, y1, x2, y2 = box.tolist()
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                object_coords.append([x_center, y_center])

        num_objects = len(object_coords)

        validation_data.append({
            "sequence_id": sequence_id,
            "frame": frame,
            "num_objects": num_objects,
            "object_coords": object_coords
        })

    with open("my_anno.json", "w") as f:
        json.dump(validation_data, f, indent=4)

    print("my_anno.json has been saved")