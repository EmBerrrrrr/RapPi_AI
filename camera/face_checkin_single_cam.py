import cv2
import time
import requests
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_recognition.face_detection import FaceDetector
from face_recognition.face_recognition import FaceRecognizer


class FaceCheckIn:
    def __init__(self, cam_id=0):
        print("INIT FACE CHECK-IN SYSTEM")

        # Camera
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # AI
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()

        # API
        self.API_URL = "https://famous-kodiak-delicate.ngrok-free.app/api/v1/work-shifts/face-check-in"

        # chống spam
        self.last_checkin_time = 0
        self.cooldown = 10  # seconds

        print("System ready!")

    def run(self):
        print("Camera started... (press Q to quit)")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            display = frame.copy()

            face_detected = False
            embedding = None

            # ===== DETECT FACE =====
            try:
                faces, boxes = self.face_detector.extract_all_faces(frame)

                if len(faces) > 0:
                    face_detected = True
                    face_img = faces[0]

                    embedding = self.face_recognizer.get_embedding(face_img)

                    # draw box
                    display = self.face_detector.draw_faces(display)

            except Exception as e:
                print("Face error:", e)

            # ===== CALL API =====
            if face_detected and embedding is not None:
                now = time.time()

                if now - self.last_checkin_time > self.cooldown:
                    print("Face detected → sending...")

                    self.call_api(embedding)

                    self.last_checkin_time = now

            # ===== UI =====
            status = "NO FACE"
            if face_detected:
                status = "FACE DETECTED"

            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("FACE CHECK-IN", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def call_api(self, embedding):
        try:
            res = requests.post(
                self.API_URL,
                json={
                    "embedding": embedding.tolist()
                },
                timeout=5
            )

            if res.status_code == 200:
                data = res.json()
                print("RESPONSE:", data)
            else:
                print("API ERROR:", res.text)

        except Exception as e:
            print("CALL API FAIL:", e)

    def cleanup(self):
        print("Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = FaceCheckIn(cam_id=0)
    app.run()