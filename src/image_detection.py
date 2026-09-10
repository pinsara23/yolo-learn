from pathlib import Path
from ultralytics import YOLO

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_ROOT / "images" / "test.jpg"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "detected_image.jpg"

# load model
model = YOLO("yolov8s-world.pt")

#run detections
results = model(IMAGE_PATH)

#show and save results

result = results[0]

result.show()

result.save(filename=str(OUTPUT_PATH))
print(f"Detection completed.")
print(f"Result saved to: {OUTPUT_PATH}")
