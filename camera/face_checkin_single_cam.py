import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import cv2
import time
import requests
import sys
import os
# Đường dẫn API Backend (.NET)
# Thay localhost bằng IP của máy chạy Backend nếu chạy từ thiết bị khác
API_BASE_URL = "https://localhost:7015" 
CHECKIN_URL = f"{API_BASE_URL}/api/v1/work-shifts/face-check-in"
CHECKOUT_URL = f"{API_BASE_URL}/api/v1/work-shifts/face-check-out"

def capture_and_process(mode='checkin', token=None):
    """
    Hàm chụp 1 ảnh duy nhất và gửi về Backend
    mode: 'checkin' hoặc 'checkout'
    token: JWT Token của nhân viên (nếu có)
    """
    print(f"--- Đang khởi động hệ thống nhận diện {mode.upper()} ---")
    
    # 1. Mở Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("LỖI: Không thể mở Camera")
        return False

    url = CHECKIN_URL if mode == 'checkin' else CHECKOUT_URL
    start_time = time.time()
    captured_frame = None
    
    print("Đang quét khuôn mặt... Vui lòng nhìn thẳng vào camera.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Hiển thị cửa sổ xem trước (Tùy chọn - có thể ẩn đi trên thiết bị nhúng)
            cv2.imshow("Scan Face", frame)
            
            # Tự động chụp sau 5 giây để đảm bảo camera đã lấy nét xong
            if time.time() - start_time > 5.0:
                captured_frame = frame
                break

            # Nhấn 'q' để hủy
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Đã hủy quét.")
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # 2. Gửi ảnh về Backend (C#)
    if captured_frame is not None:
        print("Đang gửi ảnh về hệ thống đối soát...")
        try:
            _, img_encoded = cv2.imencode('.jpg', captured_frame)
            files = {
                'FaceImage': ('checkin.jpg', img_encoded.tobytes(), 'image/jpeg')
            }
            
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            
            response = requests.post(url, files=files, headers=headers, timeout=15, verify=False)
            
            if response.status_code == 200:
                print("✅ THÀNH CÔNG:", response.json().get('message', 'Điểm danh hoàn tất'))
                return True
            else:
                print("❌ THẤT BẠI:", response.text)
                return False
                
        except Exception as e:
            print(f"❌ LỖI KẾT NỐI BE: {e}")
            return False
    
    return False

if __name__ == "__main__":
    # Cách dùng: python face_checkin_single_cam.py [checkin/checkout] [JWT_TOKEN]
    mode = sys.argv[1] if len(sys.argv) > 1 else 'checkin'
    token = sys.argv[2] if len(sys.argv) > 2 else None
    
    capture_and_process(mode, token)
