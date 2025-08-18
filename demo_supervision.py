import cv2
import supervision as sv
from ultralytics import YOLO

# Modelo YOLOv8
model = YOLO("yolov8n.pt")
box_annotator = sv.BoxAnnotator()

# Captura de webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    cv2.imshow("Supervision + YOLOv8", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
