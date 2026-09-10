from pathlib import Path
from ultralytics import YOLOWorld

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_ROOT / "images" / "test.jpg"

TARGET_CLASS = "person"
CONFIDENCE_THRESHOLD = 0.70

model = YOLOWorld("yolov8s-world.pt")

results = model(IMAGE_PATH)
result = results[0]

target_found = False

for box, in result.boxes:

    class_id = int(box.cls[0])
    clas_name = model.names[class_id]

    confidence = float(box.conf[0])

    if clas_name != TARGET_CLASS:
        continue

    if confidence < CONFIDENCE_THRESHOLD:
        continue

    x1, y1, x2, y2 = box.xyxy[0].tolist()
    center_X = (x1 + x2)/2
    center_Y = (y1 + y2)/2

    print("\n Person detected")

    print(
        f"Confidence: "
        f"{confidence:.2%}"
    )

    print(
        f"Center: "
        f"({center_X:.0f}, {center_Y:.0f})"
    )

    target_found = True

if not target_found:
    print(f"\nNo {TARGET_CLASS} detected with confidence above {CONFIDENCE_THRESHOLD:.2%}")
