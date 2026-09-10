from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_ROOT / "images" / "test.jpg"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "detected_image.jpg"

model = YOLO("yolov8s-world.pt")

results = model(IMAGE_PATH)

#get 1st result
result = results[0]

print(f"Detections")

#go through each detection and print the class name and confidence score
for index, box in enumerate(result.boxes, start=1):

    #class id
    class_id = int(box.cls[0])

    #convert class id to readable format
    class_name = model.names[class_id]

    #confidence score
    confidence_score = float(box.conf[0])

    #bounding box coordinates
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    print(f"Detection #{index}")

    print(f"Class ID: {class_id}")
    print(f"Class Name: {class_name}")

    print(f"Confidence: {confidence_score:.2f}")

    print(
        f"Bounding Box:"
        f" ({x1:.0f}, {y1:.0f})"
        f" -> ({x2:.0f}, {y2:.0f})"
    )

    print("---------------------------------")


print(f"\nTotal objects detected: {len(result.boxes)}")