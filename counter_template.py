import cv2
import numpy as np
import time

# Imagen de referencia inicial (puede estar vacía si aún no capturaste nada)
template = cv2.imread("snapshot_1758818418.png", cv2.IMREAD_GRAYSCALE)
if template is not None:
    w, h = template.shape[::-1]
else:
    template = None

# Umbral de similitud (ajustar según pruebas)
THRESHOLD = 0.8

# Stream (ejemplo con RTSP / OBS + MediaMTX)
url = "rtsp://localhost:8554/live/test"
cap = cv2.VideoCapture(url)

count = 0
detected = False  # Para no contar frames repetidos seguidos

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer frame del stream")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Solo hacer matching si ya tenés un template cargado
    if template is not None:
        # cv2.matchTemplate es un algoritmo de OpenCV que compara cada región de la imagen del template
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= THRESHOLD)

        if len(loc[0]) > 0:
            if not detected:  # Solo cuenta la primera vez que aparece
                count += 1
                print(f"Pantalla detectada {count} veces")
                detected = True
        else:
            detected = False

    # Overlay del contador (verdecito)
    cv2.putText(frame, f"Contador: {count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Stream", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):  # Guardar snapshot
        filename = f"snapshot_{int(time.time())}.png"
        cv2.imwrite(filename, gray)  # Guarda en escala de grises
        print(f"Snapshot guardado como {filename}")

cap.release()
cv2.destroyAllWindows()
