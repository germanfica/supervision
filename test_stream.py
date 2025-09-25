# test_stream.py
import cv2

url = "rtsp://localhost:8554/live/test"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Failed to open the stream:", url)
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("Stream Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
