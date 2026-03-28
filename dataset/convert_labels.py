import json
from pathlib import Path

with open('../true_labels.json', 'r') as f:
    labels = json.load(f)

IMG_WIDTH = 640
IMG_HEIGHT = 512
CLASS_ID = 0
BBOX_WIDTH = 20
BBOX_HEIGHT = 20

for label in labels:
    sequence_id = label['sequence_id']
    frame = label['frame']
    num_objects = label['num_objects']
    object_coords = label['object_coords']

    label_dir = Path(f'D:\\Dataset\\train\\{sequence_id}')
    label_dir.mkdir(parents=True, exist_ok=True)
    label_file = label_dir / f'{frame}.txt'

    with open(label_file, 'w') as f:
        for coord in object_coords:
            x_center, y_center = coord
            x_norm = x_center / IMG_WIDTH
            y_norm = y_center / IMG_HEIGHT
            w_norm = BBOX_WIDTH / IMG_WIDTH
            h_norm = BBOX_HEIGHT / IMG_HEIGHT
            f.write(f'{CLASS_ID} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n')

print("success to convert labels")
