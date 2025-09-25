import cv2
import supervision as sv
from ultralytics import YOLO

# Pretrained YOLOv8 model (examples: "yolov8n.pt", "yolov8s.pt", etc.)
model = YOLO("yolov8n.pt")
box_annotator = sv.BoxAnnotator()

# Stream URL (RTSP or HLS)
# For local testing with MediaMTX, you might use something like:
# url = "rtsp://localhost:8554/live/test"
# Capture from MediaMTX (RTSP)
url = "rtsp://localhost:8554/live/test"  # or a YouTube URL via streamlink/ffmpeg/mediamtx
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from stream")
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated = box_annotator.annotate(
        scene=frame.copy(),    # important: use a copy
        detections=detections  # without explicit labels
    )

    cv2.imshow("Supervision + YOLOv8", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
