import os
import cv2
import numpy as np
import requests

ESP_IP = "192.168.0.115" \
""      # <- your ESP32's IP from the serial monitor
OUT_DIR = "dataset"
os.makedirs(OUT_DIR, exist_ok=True)

def fetch(path):
    r = requests.get(f"http://{ESP_IP}/{path}", timeout=10)
    r.raise_for_status()
    return r.content          # raw JPEG bytes

def next_index():
    ids = [int(f.split("_")[0]) for f in os.listdir(OUT_DIR)
           if f.endswith("_left.jpg") and f.split("_")[0].isdigit()]
    return max(ids, default=0) + 1

def decode(jpg):
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

idx = next_index()
print("SPACE = save pair   Q = quit")

while True:
    try:
        left_jpg = fetch("cam_left")
        right_jpg = fetch("cam_right")
    except requests.RequestException as e:
        print("Fetch failed:", e)
        continue

    preview = np.hstack([decode(left_jpg), decode(right_jpg)])
    cv2.putText(preview, f"next: {idx}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("left | right", preview)

    key = cv2.waitKey(50) & 0xFF
    if key == ord(' '):
        # Write the exact bytes shown in the preview, so there's no re-compression
        with open(f"{OUT_DIR}/{idx}_left.jpg", "wb") as f:
            f.write(left_jpg)
        with open(f"{OUT_DIR}/{idx}_right.jpg", "wb") as f:
            f.write(right_jpg)
        print(f"Saved pair {idx}")
        idx += 1
    elif key == ord('q'):
        break

cv2.destroyAllWindows()