import requests
import numpy as np
import cv2

class IPCamera:
    def __init__(self, url):
        self.url = url

    def get_frame(self):
        for _ in range(3): 
            try:
                response = requests.get(self.url, timeout=2, stream=True)

                if response.status_code == 200:
                    img_arr = np.frombuffer(response.content, np.uint8)
                    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                    if frame is None:
                        print(f"[DECODE FAIL] {self.url}")
                        continue

                    return frame

            except Exception as e:
                print(f"[REQUEST FAIL] {self.url} - {e}")

        return None