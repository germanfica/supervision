import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Use DSHOW for Windows compatibility

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Usar el método built-in de Ultralytics para dibujar pose
    annotated = results[0].plot()

    cv2.imshow("YOLOv8 Pose", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
