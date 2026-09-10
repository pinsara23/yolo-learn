from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_ROOT / "images" / "test.jpg"


model = YOLO("yolov8s-world.pt")


results = model(IMAGE_PATH)

result = results[0]


for box in result.boxes:

    class_id = int(box.cls[0])

    class_name = model.names[class_id]

    confidence = float(box.conf[0])

    x1, y1, x2, y2 = box.xyxy[0].tolist()


    # Calculate center point
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2


    print(f"\nObject: {class_name}")

    print(
        f"Confidence: "
        f"{confidence:.2%}"
    )

    print(
        f"Center: "
        f"({center_x:.0f}, {center_y:.0f})"
    )