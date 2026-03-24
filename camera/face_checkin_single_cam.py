import cv2
import time
import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_recognition.face_detection import FaceDetector
from face_recognition.face_recognition import FaceRecognizer


class FaceCheckIn:
    def __init__(self, cam_id=0):
        print("INIT FACE SCAN")

        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()

        self.API_URL = "https://famous-kodiak-delicate.ngrok-free.app/api/v1/work-shifts/face-check-in"

    def run(self):
        print("Opening camera...")

        start_time = time.time()
        timeout = 5  # chạy đủ 5s

        best_embedding = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            try:
                faces, _ = self.face_detector.extract_all_faces(frame)

                if len(faces) > 0:
                    face_img = faces[0]

                    embedding = self.face_recognizer.get_embedding(face_img)

                    # luôn update embedding mới nhất trong 10s
                    best_embedding = embedding

                    print(f"Face detected (vector {len(embedding)})")

            except Exception as e:
                print("Error:", e)

            cv2.imshow("SCAN FACE", frame)

            # đủ 5s thì dừng
            if time.time() - start_time > timeout:
                print("Finished scanning (5s)")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # sau 5s mới gửi
        if best_embedding is not None:
            print("Sending embedding...")
            self.send_embedding(best_embedding)
            print("Done")
        else:
            print("No face detected")

        self.cleanup()

    def send_embedding(self, embedding):
        try:
            requests.post(
                self.API_URL,
                json={"embedding": embedding.tolist()},
                timeout=3
            )
        except Exception as e:
            print("Send fail:", e)

    def cleanup(self):
        print("Closing camera...")
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = FaceCheckIn()
    app.run()