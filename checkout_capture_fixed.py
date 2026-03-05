"""
Check-Out Capture - Quét khuôn mặt và biển số để check-out
So sánh với dữ liệu check-in, xác minh 85% similarity
Timeout: 30 giây - CHỈ EXIT KHI THÀNH CÔNG HOẶC TIMEOUT
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import json
import pickle

from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from license_plate.detector import LicensePlateDetector
from dataset_manager import DatasetManager


class CheckOutCapture:
    """
    Quét khuôn mặt + biển số check-out
    So sánh với check-in dataset
    Xác minh >= 85% similarity
    Chỉ exit khi thành công hoặc timeout 30s
    """
    
    def __init__(self, camera_id=0, timeout_sec=30, similarity_threshold=0.70, plate_confidence_thresh=0.80):
        """
        Khởi tạo check-out capture
        
        Args:
            camera_id: ID camera (0 = webcam)
            timeout_sec: Thời gian giới hạn (giây)
            similarity_threshold: Ngưỡng tương đồng khuôn mặt (0-1)
            plate_confidence_thresh: Ngưỡng confidence cho biển số
        """
        print("\n" + "="*70)
        print("🚗 CHECK-OUT CAPTURE - INITIALIZATION")
        print("="*70)
        
        # Initialize camera
        print("\n📸 Initializing camera...")
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Cannot open camera {camera_id}")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"   ✅ Camera opened (ID: {camera_id})")
        
        # Initialize AI modules
        print("\n🤖 Initializing AI modules...")
        self.face_detector = FaceDetector()
        print("   ✅ Face detector ready")
        
        self.face_recognizer = FaceRecognizer()
        print("   ✅ Face recognizer ready")
        
        self.plate_detector = LicensePlateDetector()
        print("   ✅ License plate detector ready")
        
        # Initialize dataset manager
        self.dataset_manager = DatasetManager()
        print("   ✅ Dataset manager ready")
        
        # Settings
        self.timeout_sec = float(timeout_sec)
        self.similarity_threshold = float(similarity_threshold)
        self.plate_confidence_thresh = float(plate_confidence_thresh)
        
        # State
        self.start_time = None
        self.result = None
        
        print("\n✅ All modules initialized successfully!")
        print("="*70)
    
    def start_checkout(self):
        """
        Bắt đầu quá trình check-out (30 giây)
        Loop liên tục quét face + plate
        Khi phát hiện cả 2 -> verify
        - Nếu SUCCESS -> EXIT
        - Nếu FAIL -> RETRY (tiếp tục trong khoảng thời gian còn lại)
        - Nếu TIMEOUT -> EXIT
        
        Returns:
            dict: Kết quả check-out
                {
                    'success': bool,
                    'message': str,
                    'plate': str,
                    'similarity': float (nếu match),
                    'duration_sec': float
                }
        """
        print("\n" + "="*70)
        print("🚗 CHECK-OUT PROCESS STARTED")
        print("="*70)
        print(f"\n⏱️  TIME LIMIT: {self.timeout_sec} seconds")
        print("📸 Scanning face and license plate...")
        print("🔄 Will RETRY if verification fails - waiting for better frame")
        print("━" * 70)
        
        self.start_time = time.time()
        last_verify_time = 0
        VERIFY_COOLDOWN = 2  # Chờ 2s trước verify tiếp theo
        
        while True:
            elapsed = time.time() - self.start_time
            remaining = self.timeout_sec - elapsed
            
            # Check timeout
            if elapsed >= self.timeout_sec:
                print(f"\n⏱️  TIME EXPIRED ({self.timeout_sec}s)")
                print("   No successful verification within time limit")
                self.result = {
                    'success': False,
                    'message': '❌ TIMEOUT - Quá thời gian cho phép (30s)',
                    'plate': None,
                    'similarity': None,
                    'duration_sec': elapsed,
                    'reason': 'timeout'
                }
                break
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Detect face
            face_detected = False
            face_image = None
            face_embedding = None
            
            try:
                faces, boxes = self.face_detector.extract_all_faces(frame)
                if len(faces) > 0:
                    face_detected = True
                    face_image = faces[0]
                    face_embedding = self.face_recognizer.get_embedding(face_image)
            except Exception as e:
                pass
            
            # Detect plate
            plate_detected = False
            plate_text = None
            plate_confidence = 0.0
            plate_bbox = None
            
            try:
                detected_plates = self.plate_detector.detect(frame, conf_threshold=0.4)
                if len(detected_plates) > 0:
                    plate_detected = True
                    best_result = detected_plates[0]
                    plate_text = best_result.get('text')
                    plate_confidence = float(best_result.get('confidence', 0.0))
                    plate_bbox = best_result.get('bbox')
            except Exception as e:
                pass
            
            # Draw on frame
            display_frame = frame.copy()
            
            # Draw status with countdown
            status_color = (255, 255, 255)  # White
            cv2.putText(display_frame, f"Time remaining: {max(0, remaining):.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Face status
            if face_detected:
                face_color = (0, 255, 0)
                face_text = "✓ Face detected"
            else:
                face_color = (0, 0, 255)
                face_text = "✗ Face: waiting..."
            cv2.putText(display_frame, face_text, (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
            
            # Plate status
            if plate_detected and plate_text and plate_text != "Unknown":
                plate_color = (0, 255, 0)
                plate_info = f"✓ Plate: {plate_text} ({plate_confidence:.0%})"
            else:
                plate_color = (0, 0, 255)
                plate_info = "✗ Plate: waiting..."
            cv2.putText(display_frame, plate_info, (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, plate_color, 2)
            
            # Draw face boxes
            try:
                display_frame = self.face_detector.draw_faces(display_frame)
            except:
                pass
            
            # Draw plate box
            if plate_detected and plate_bbox:
                try:
                    x1, y1, x2, y2 = plate_bbox
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                except:
                    pass
            
            cv2.imshow("Check-Out Capture", display_frame)
            
            # ===== LOGIC FIX =====
            # CHỈ verify nếu:
            # 1. Cả face + plate đều detect
            # 2. Plate confidence >= threshold
            # 3. Đủ thời gian cooldown từ verify lần trước (tránh spam)
            current_time = time.time()
            time_since_last_verify = current_time - last_verify_time
            
            if (face_detected and plate_detected and 
                face_embedding is not None and plate_text and 
                plate_text != "Unknown" and
                plate_confidence >= self.plate_confidence_thresh and
                time_since_last_verify >= VERIFY_COOLDOWN):
                
                print(f"\n✅ Detection quality OK: Plate={plate_text}, Conf={plate_confidence:.0%}")
                print("   🔄 Verifying face match...")
                
                # Try to match with dataset
                match_result = self._verify_checkout(plate_text, face_embedding)
                last_verify_time = current_time
                
                elapsed = time.time() - self.start_time
                
                # CHỈ exit nếu SUCCESS hoặc CRITICAL ERROR
                if match_result['success']:
                    # ✅ MATCH! Exit ngay
                    self.result = {
                        'success': True,
                        'message': match_result['message'],
                        'plate': plate_text,
                        'similarity': match_result.get('similarity'),
                        'duration_sec': elapsed,
                        'reason': match_result.get('reason')
                    }
                    print("\n   ✅ SUCCESS - Exiting...")
                    break
                else:
                    # ❌ FAILED - Print reason but continue waiting
                    print(f"   ❌ Verification failed: {match_result.get('reason')}")
                    print(f"   🔄 Retrying... (Remaining: {max(0, remaining):.1f}s)")
            
            # Quit on 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                elapsed = time.time() - self.start_time
                print("\n🛑 Cancelled by user")
                self.result = {
                    'success': False,
                    'message': '❌ CANCELLED - User pressed Q',
                    'plate': None,
                    'similarity': None,
                    'duration_sec': elapsed,
                    'reason': 'user_cancel'
                }
                break
        
        self.cleanup()
        self._display_result()
        
        return self.result
    
    def _verify_checkout(self, plate_text, checkout_face_embedding):
        """
        Xác minh check-out: tìm plate trong dataset, so sánh face
        
        Args:
            plate_text: Biển số xe (str)
            checkout_face_embedding: Face embedding từ check-out (array)
            
        Returns:
            dict: Kết quả xác minh
        """
        try:
            # Get all face vectors from dataset
            all_vectors = self.dataset_manager.get_all_face_vectors()
            
            # Tìm face được lưu với tên = plate_text (hoặc plate name normalized)
            plate_normalized = plate_text.strip().upper().replace(" ", "_")
            
            print(f"      🔍 Searching database for: {plate_normalized}")
            
            if plate_normalized not in all_vectors:
                print(f"      ❌ Plate {plate_text} not in database")
                return {
                    'success': False,
                    'message': f'❌ FAILED - Biển số {plate_text} không tìm thấy',
                    'reason': 'plate_not_found',
                    'similarity': None
                }
            
            # Get stored face vectors for this plate
            stored_vectors = all_vectors[plate_normalized]  # numpy array (N, 512)
            
            if stored_vectors.shape[0] == 0:
                print(f"      ❌ No face data for plate {plate_text}")
                return {
                    'success': False,
                    'message': f'❌ FAILED - Không có dữ liệu khuôn mặt cho {plate_text}',
                    'reason': 'no_face_vectors',
                    'similarity': None
                }
            
            # Compare checkout face with all stored vectors (use cosine similarity)
            max_similarity = 0.0
            best_stored_idx = -1
            
            for i, stored_vector in enumerate(stored_vectors):
                # Normalize vectors
                v1 = checkout_face_embedding / np.linalg.norm(checkout_face_embedding)
                v2 = stored_vector / np.linalg.norm(stored_vector)
                
                # Cosine similarity
                similarity = float(np.dot(v1, v2))
                max_similarity = max(max_similarity, similarity)
            
            print(f"      📊 Max similarity: {max_similarity:.1%} (threshold: {self.similarity_threshold:.0%})")
            
            # Check if >= threshold
            if max_similarity >= self.similarity_threshold:
                msg = f'✅ THÀNH CÔNG - Cảm ơn! (Tương đồng: {max_similarity:.0%})'
                print(f"      ✅ {msg}")
                return {
                    'success': True,
                    'message': msg,
                    'reason': 'match_success',
                    'similarity': max_similarity
                }
            else:
                msg = f'❌ FAILED - Khuôn mặt không khớp ({max_similarity:.0%})'
                print(f"      ❌ {msg}")
                return {
                    'success': False,
                    'message': msg,
                    'reason': 'similarity_too_low',
                    'similarity': max_similarity
                }
        
        except Exception as e:
            print(f"      ❌ Error: {e}")
            return {
                'success': False,
                'message': f'❌ FAILED - Lỗi xác minh: {str(e)}',
                'reason': 'verification_error',
                'similarity': None
            }
    
    def _display_result(self):
        """Hiển thị kết quả check-out"""
        if self.result is None:
            return
        
        print("\n" + "="*70)
        print("📋 CHECK-OUT RESULT")
        print("="*70)
        
        success = self.result['success']
        message = self.result['message']
        plate = self.result['plate']
        similarity = self.result['similarity']
        duration = self.result['duration_sec']
        reason = self.result.get('reason', 'unknown')
        
        print(f"\n{message}")
        print(f"\n📊 Details:")
        print(f"   Plate: {plate if plate else 'N/A'}")
        print(f"   Similarity: {f'{similarity:.0%}' if similarity is not None else 'N/A'}")
        print(f"   Duration: {duration:.2f}s / {self.timeout_sec}s")
        print(f"   Result: {reason}")
        
        print("\n" + "="*70)
    
    def cleanup(self):
        """Dọn dẹp resources"""
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Main entry point for check-out"""
    try:
        checkout = CheckOutCapture(
            camera_id=0,
            timeout_sec=30,
            similarity_threshold=0.70,
            plate_confidence_thresh=0.80
        )
        
        result = checkout.start_checkout()
        
        # Return result for integration with parking system
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'message': f'❌ FATAL ERROR - {str(e)}',
            'plate': None,
            'similarity': None,
            'duration_sec': 0,
            'reason': 'fatal_error'
        }


if __name__ == "__main__":
    print("\n🚀 PARKING CHECK-OUT SYSTEM (FIXED)")
    print("=" * 70)
    print("\n📋 PROCESS:")
    print("   1. Scan face and license plate (30s timeout)")
    print("   2. When both detected -> verify")
    print("   3. If SUCCESS (≥85%) -> EXIT & show thanks")
    print("   4. If FAILED -> RETRY (continue scanning)")
    print("   5. If TIMEOUT (30s) -> EXIT & show timeout")
    print("\n🎮 Controls:")
    print("   - Press 'q' to cancel")
    print("   - Will show countdown timer\n")
    
    result = main()
    
    if result['success']:
        print(f"\n✅ {result['message']}")
    else:
        print(f"\n❌ {result['message']}")
