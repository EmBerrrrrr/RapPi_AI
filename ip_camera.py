import cv2

class IPCamera:
    def __init__(self, url, username="admin", password="admin"):
        # 🔥 nhúng auth vào URL
        if "@" not in url:
            url = url.replace("http://", f"http://{username}:{password}@")

        self.url = url

        # 🔥 dùng FFMPEG để đọc MJPEG
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            print(f"Cannot open camera: {self.url}")
        else:
            print(f"Camera connected: {self.url}")

    def get_frame(self):
        if not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()

        if not ret:
            print(f"[FRAME FAIL] {self.url}")
            return None

        return frame