import cv2
from ultralytics import YOLOWorld

CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.50

model = YOLOWorld("yolov8s-world.pt")

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError(f"Could not open camera with index {CAMERA_INDEX}")

print("YOLO detection is running \n Press 'q' to exit")

while True:
    success, frame = camera.read()
    if not success:
        break

    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]


    #read individual detections from the result
    for box, in result.boxes:

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_X = int((x1 + x2)/2)
        center_Y = int((y1 + y2)/2)

        print(
            f"{class_name:<15} "
            f"{confidence:.2%} "
            f"center=({center_X}, {center_Y})"
        )

        #display detections
        cv2.imshow("YOLO Detection", result.plot())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


camera.release()

cv2.destroyAllWindows()