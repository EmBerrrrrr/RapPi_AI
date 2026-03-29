import sys
import cv2
import time
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===== INPUT =====
mode = sys.argv[1] if len(sys.argv) > 1 else "checkin"
token = sys.argv[2] if len(sys.argv) > 2 else ""

print("[START]", mode)

# ===== CONFIG =====
API_BASE_URL = "https://rare-flounder-exactly.ngrok-free.app"
CHECKIN_URL = f"{API_BASE_URL}/api/v1/work-shifts/face-check-in"
CHECKOUT_URL = f"{API_BASE_URL}/api/v1/work-shifts/face-check-out"

url = CHECKIN_URL if mode == "checkin" else CHECKOUT_URL
print("[URL]", url)

# ===== LOAD FACE MODEL =====
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ===== CAMERA =====
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    sys.exit(1)

print("[INFO] Waiting for stable face (5s)...")

best_frame = None
best_face = None
best_face_size = 0
start_detect_time = None

while True:
    ret, img = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        if start_detect_time is None:
            start_detect_time = time.time()
            print("[INFO] Face detected -> start 5s timer")

        elapsed = time.time() - start_detect_time
        remaining = max(0, 5 - int(elapsed))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

            if w < 120 or h < 120:
                continue

            face_size = w * h
            if face_size > best_face_size:
                best_face_size = face_size
                best_frame = img.copy()
                best_face = (x, y, w, h)

        cv2.putText(img, f"Hold still {remaining}s", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if elapsed >= 5:
            print("[OK] Face stable for 5s")
            break

    else:
        start_detect_time = None
        best_face_size = 0
        best_frame = None
        best_face = None

        cv2.putText(img, "No face", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Face Detect", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

if best_frame is None or best_face is None:
    print("ERROR: No valid face")
    sys.exit(1)

print("[OK] Best frame selected")

# ===== CROP FACE + GIỮ BACKUP FULL FRAME =====
x, y, w, h = best_face

# 👉 crop có margin (quan trọng)
margin = 0.3
x1 = max(0, int(x - w * margin))
y1 = max(0, int(y - h * margin))
x2 = min(best_frame.shape[1], int(x + w * (1 + margin)))
y2 = min(best_frame.shape[0], int(y + h * (1 + margin)))

face_crop = best_frame[y1:y2, x1:x2]

# 👉 resize nhẹ (giữ giống BE)
face_crop = cv2.resize(face_crop, (224, 224))

cv2.imwrite("debug_face.jpg", face_crop)

# ===== ENCODE (GIỐNG CODE CŨ) =====
_, img_encoded = cv2.imencode('.jpg', face_crop)

files = {
    'FaceImage': ('checkin.jpg', img_encoded.tobytes(), 'image/jpeg')  # ⚠️ giữ nguyên tên
}

headers = {}
if token:
    headers["Authorization"] = f"Bearer {token}"

# ===== CALL BE =====
try:
    print("[SEND] Sending to BE...")

    response = requests.post(
        url,
        files=files,
        headers=headers,
        timeout=15,
        verify=False
    )

    print("[STATUS]", response.status_code)
    print("[TEXT]", response.text)

    if response.status_code == 200:
        data = response.json()
        if data.get("is_success") is True:
            print("MATCH SUCCESS")
            sys.exit(0)

    print("MATCH FAIL")
    sys.exit(1)

except Exception as e:
    print("ERROR:", e)
    sys.exit(1)