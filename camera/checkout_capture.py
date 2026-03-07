"""
Check-Out Capture - Quét khuôn mặt và biển số để check-out
So sánh với dữ liệu check-in, xác minh 85% similarity
Timeout: 30 giây
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import json
import pickle
import difflib

from face_recognition.face_detection import FaceDetector
from face_recognition.face_recognition import FaceRecognizer
from license_plate.detector import LicensePlateDetector
from dataset_manager import DatasetManager
from mqtt_client import send_checkout
from datetime import datetime, timezone


class CheckOutCapture:
    """
    Quét khuôn mặt + biển số check-out
    So sánh với check-in dataset
    Xác minh >= 70% similarity
    """
    
    def __init__(self, face_cam_id=1, plate_cam_id=0, timeout_sec=60, similarity_threshold=0.70, plate_confidence_thresh=0.80,):
        """
        Khởi tạo check-out capture
        Args:
            face_cam_id: ID camera quét khuôn mặt (0 = webcam)
            plate_cam_id: ID camera quét biển số (1 = Iriun Webcam)
            timeout_sec: Thời gian giới hạn (giây)
            similarity_threshold: Ngưỡng tương đồng khuôn mặt (0-1)
            plate_confidence_thresh: Ngưỡng confidence cho biển số
        """
        print("\n" + "="*70)
        print("🚗 CHECK-OUT CAPTURE - INITIALIZATION")
        print("="*70)
        
        # Initialize camera
        print("\n📸 Initializing camera...")
        self.face_cap = cv2.VideoCapture(face_cam_id)
        self.plate_cap = cv2.VideoCapture(plate_cam_id)
        
        if not self.face_cap.isOpened():
            raise RuntimeError(f"❌ Cannot open face camera {face_cam_id}")
        if not self.plate_cap.isOpened():
            raise RuntimeError(f"❌ Cannot open plate camera {plate_cam_id}")
        
        # Set camera resolution
        self.face_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.face_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.face_cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.plate_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.plate_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.plate_cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"   ✅ Camera opened (ID: {face_cam_id} for face, {plate_cam_id} for plate)")
        
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
        self.checkout_plate = None
        self.checkout_face_embedding = None
        self.result = None
        
        print("\n✅ All modules initialized successfully!")
        print("="*70)

        self.verify_plate_text = None
        self.verify_start_time = None
        self.verify_wait_sec = 5.0   # ⏳ chờ 5 giây để xác thực

    def start_checkout(self):
        """
        Bắt đầu quá trình check-out
        Thời gian giới hạn: 30 giây
        
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
        print("━" * 70)
        
        self.start_time = time.time()
        checkout_success = False
        VERIFY_COOLDOWN = 2
        last_verify_time = 0
        
        while True:
            elapsed = time.time() - self.start_time
            remaining = self.timeout_sec - elapsed
            
            # Check timeout
            if elapsed >= self.timeout_sec:
                print(f"\n⏱️  TIME EXPIRED ({self.timeout_sec}s)")
                self.result = {
                    'success': False,
                    'message': '❌ TIMEOUT - Quá thời gian cho phép',
                    'plate': None,
                    'similarity': None,
                    'duration_sec': elapsed,
                    'reason': 'timeout'
                }
                break
            
            # Read frame
            ret_face, face_frame = self.face_cap.read()
            ret_plate, plate_frame = self.plate_cap.read()

            if not ret_face or not ret_plate:
                print("❌ Camera read error")
                break

            # Detect face
            face_detected = False
            face_image = None
            face_embedding = None
            
            try:
                faces, boxes = self.face_detector.extract_all_faces(face_frame)
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
                detected_plates = self.plate_detector.detect(
                    plate_frame,
                    conf_threshold=self.plate_confidence_thresh
                )
                if len(detected_plates) > 0:
                    plate_detected = True
                    best_result = detected_plates[0]
                    plate_text = best_result.get('text')
                    plate_confidence = float(best_result.get('confidence', 0.0))
                    plate_bbox = best_result.get('bbox')
                    
                    plate_image = None

                    if plate_bbox and len(plate_bbox) == 4:
                        x1, y1, x2, y2 = plate_bbox

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(plate_frame.shape[1], x2), min(plate_frame.shape[0], y2)

                        if x2 > x1 and y2 > y1:
                            plate_image = plate_frame[y1:y2, x1:x2]
            except Exception as e:
                pass
            
            # Draw on frame
            face_display = face_frame.copy()
            plate_display = plate_frame.copy()

            # Draw status
            status_color = (255, 255, 255)  # White
            cv2.putText(face_display, f"Remaining: {max(0, remaining):.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            if face_detected:
                cv2.putText(face_display, "Face: OK", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(face_display, "Face: NOT FOUND", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if plate_detected and plate_text and plate_text != "Unknown":
                cv2.putText(plate_display, f"Plate: {plate_text} ({plate_confidence:.2f})", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(plate_display, "Plate: NOT FOUND", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw face boxes
            try:
                face_display = self.face_detector.draw_faces(face_display)
            except:
                pass
            
            # Draw plate box
            if plate_detected and plate_bbox:
                try:
                    x1, y1, x2, y2 = plate_bbox
                    cv2.rectangle(plate_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                except:
                    pass
            
            cv2.imshow("Check-Out Capture - Face", face_display)
            plate_display_small = cv2.resize(plate_display, (640, 480))
            cv2.imshow("Check-Out Capture - Plate", plate_display_small)
            
            # If both detected and quality OK -> verify (with cooldown)
            current_time = time.time()
            if not plate_detected:
                self.verify_plate_text = None

            if face_detected and plate_detected and face_embedding is not None and plate_text and plate_text != "Unknown":

                if self.verify_plate_text != plate_text:
                    self.verify_plate_text = plate_text
                    self.verify_start_time = current_time
                    print(f"⏳ Plate detected ({plate_text}) — waiting {self.verify_wait_sec}s to stabilize...")

                else:
                    elapsed_verify = current_time - self.verify_start_time

                    if elapsed_verify >= self.verify_wait_sec and current_time - last_verify_time > VERIFY_COOLDOWN:

                        print(f"\n✅ Plate & Face stable for {self.verify_wait_sec}s")
                        print("🔄 Verifying against database...")

                        match_result = self._verify_checkout(plate_text, face_embedding)

                        last_verify_time = current_time
                        elapsed_total = time.time() - self.start_time

                        if match_result['success']:

                            checkout_info = self.dataset_manager.record_checkout(plate_text)

                            self.result = {
                                'success': True,
                                'message': match_result['message'],
                                'plate': plate_text,
                                'similarity': match_result.get('similarity'),
                                'duration_sec': elapsed_total,
                                'reason': match_result.get('reason')
                            }

                            if checkout_info:
                                self.result['time_in'] = checkout_info.get('time_in')
                                self.result['time_out'] = checkout_info.get('time_out')
                                self.result['parking_duration_sec'] = checkout_info.get('duration_sec')

                            #Sent MQTT checkout event
                            try:
                                send_checkout(
                                    plate_number=plate_text,
                                    similarity=match_result.get('similarity'),
                                    face_img=face_image,
                                    plate_img=plate_image,
                                    camera_ip="192.168.1.20"
                                )

                                print("📡 MQTT checkout event sent")

                            except Exception as e:
                                print(f"⚠️ MQTT send failed: {e}")

                            break

                        else:
                            print(f"❌ Verification failed ({match_result['reason']})")
                            print(f"🔄 Retrying... Remaining: {remaining:.1f}s")
            
            # Quit on 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Cancelled by user")
                self.result = {
                    'success': False,
                    'message': '❌ CANCELLED',
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

            print(f"\n🔍 Searching for plate: {plate_normalized}")

            # Exact match first
            if plate_normalized not in all_vectors:
                # Try fuzzy/heuristic matching (common OCR confusions)
                print("   🔎 Exact match not found — trying fuzzy match...")

                def ocr_fix(s: str) -> str:
                    m = {
                        'I': '1', 'L': '1', 'O': '0', 'Q': '0',
                        'S': '5', 'Z': '2', 'B': '8', 'G': '6',
                        'I': 'K', 
                    }
                    out = []
                    for ch in s:
                        out.append(m.get(ch, ch))
                    return ''.join(out)

                candidates = list(all_vectors.keys())
                best_candidate = None
                best_score = 0.0

                for cand in candidates:
                    # compare normalized forms
                    cand_norm = str(cand).upper()
                    # try direct ratio
                    r1 = difflib.SequenceMatcher(None, plate_normalized, cand_norm).ratio()
                    # try mapping OCR confusions
                    r2 = difflib.SequenceMatcher(None, ocr_fix(plate_normalized), cand_norm).ratio()
                    r3 = difflib.SequenceMatcher(None, plate_normalized, ocr_fix(cand_norm)).ratio()
                    score = max(r1, r2, r3)
                    if score > best_score:
                        best_score = score
                        best_candidate = cand

                print(f"   🔎 Best fuzzy candidate: {best_candidate} (score={best_score:.2f})")

                # Accept candidate if reasonably close
                if best_candidate and best_score >= 0.7:
                    print(f"   ✅ Using fuzzy-matched plate: {best_candidate}")
                    plate_normalized = best_candidate
                else:
                    print(f"   ❌ Plate {plate_text} not found in database (no close match)")
                    return {
                        'success': False,
                        'message': f'❌ FAILED - Biển số {plate_text} không tìm thấy trong database',
                        'reason': 'plate_not_found',
                        'similarity': None
                    }
            
            # Get stored face vectors for this plate
            stored_vectors = all_vectors[plate_normalized]  # numpy array (N, 512)
            
            if stored_vectors.shape[0] == 0:
                print(f"   ❌ No face vectors for plate {plate_text}")
                return {
                    'success': False,
                    'message': f'❌ FAILED - Không có dữ liệu khuôn mặt cho biển số {plate_text}',
                    'reason': 'no_face_vectors',
                    'similarity': None
                }
            
            # Compare checkout face with all stored vectors (use cosine similarity)
            max_similarity = 0.0
            best_stored_idx = -1
            
            for i, stored_vector in enumerate(stored_vectors):
                # Normalize vectors
                v1 = checkout_face_embedding / (np.linalg.norm(checkout_face_embedding) + 1e-8)
                v2 = stored_vector / (np.linalg.norm(stored_vector) + 1e-8)
                
                # Cosine similarity
                similarity = float(np.dot(v1, v2))
                
                print(f"   Comparison {i+1}: {similarity:.4f}")
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_stored_idx = i
            
            print(f"\n   Max similarity: {max_similarity:.4f}")
            print(f"   Threshold: {self.similarity_threshold:.4f}")
            
            # Check if >= threshold
            if max_similarity >= self.similarity_threshold:
                print(f"\n✅ MATCH! Similarity: {max_similarity:.4f} (>= {self.similarity_threshold})")
                return {
                    'success': True,
                    'message': f'✅ THÀNH CÔNG - Cảm ơn! (Tương đồng: {max_similarity:.1%})',
                    'reason': 'match_success',
                    'similarity': max_similarity
                }
            else:
                print(f"\n❌ NO MATCH. Similarity: {max_similarity:.4f} (< {self.similarity_threshold})")
                return {
                    'success': False,
                    'message': f'❌ FAILED - Khuôn mặt không khớp (Tương đồng: {max_similarity:.1%})',
                    'reason': 'similarity_too_low',
                    'similarity': max_similarity
                }
        
        except Exception as e:
            print(f"\n❌ Error during verification: {e}")
            import traceback
            traceback.print_exc()
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
        print(f"   Similarity: {f'{similarity:.1%}' if similarity is not None else 'N/A'}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Reason: {reason}")
        
        print("\n" + "="*70)
    
    def cleanup(self):
        """Dọn dẹp resources"""
        if self.face_cap:
            self.face_cap.release()

        if self.plate_cap:
            self.plate_cap.release()
        cv2.destroyAllWindows()

def main():
    """Main entry point for check-out"""
    try:
        checkout = CheckOutCapture(
        face_cam_id=1,
        plate_cam_id=0,
        timeout_sec=60,
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
    print("\n🚀 PARKING CHECK-OUT SYSTEM")
    print("=" * 70)
    print("\n📋 PROCESS:")
    print("   1. Scan face and license plate")
    print("   2. Find matching record in database")
    print("   3. Compare face similarity (>= 70%)")
    print("   4. Show result (success/failure)")
    print("\n⏱️  TIME LIMIT: 30 seconds")
    print("🎮 Press 'q' to cancel\n")
    
    result = main()
    
    if result['success']:
        print(f"\n✅ {result['message']}")
    else:
        print(f"\n❌ {result['message']}")
