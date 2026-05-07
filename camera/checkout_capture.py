
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
from mqtt_client import register_config_callback, register_response_callback

from face_recognition.face_detection import FaceDetector
from face_recognition.face_recognition import FaceRecognizer
from license_plate.detector import LicensePlateDetector
from dataset_manager import DatasetManager
from mqtt_client import send_checkout
from datetime import datetime, timezone
from plate_utils import normalize_plate
from ip_camera import IPCamera

class CheckOutCapture:
    """
    Quét khuôn mặt + biển số check-out
    So sánh với check-in dataset
    Xác minh >= 70% similarity
    """
    
    def __init__(self,frame_skip = 2,frame_count = 0,last_embedding_time = 0, face_cam_url=None, plate_cam_url=None, timeout_sec=60, similarity_threshold=0.6, plate_confidence_thresh=0.80, last_plate_logged=None, self.last_face_seen_time = 0):
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
        print("CHECK-OUT CAPTURE - INITIALIZATION")
        print("="*70)
        
        # Initialize camera
        print("\n📸 Initializing camera...")
        self.face_cam = IPCamera(face_cam_url)
        self.plate_cam = IPCamera(plate_cam_url)

        print("Camera IP initialized")
        self.frame_count = frame_count
        self.frame_skip = frame_skip
        self.last_embedding_time = last_embedding_time
        
        print(f"    Face cam: {face_cam_url}")
        print(f"    Plate cam: {plate_cam_url}")
        
        # Initialize AI modules
        print("\nInitializing AI modules...")
        self.face_detector = FaceDetector()
        print("    Face detector ready")
        
        self.face_recognizer = FaceRecognizer()
        print("    Face recognizer ready")
        
        self.plate_detector = LicensePlateDetector()
        print("    License plate detector ready")
        
        # Initialize dataset manager
        self.dataset_manager = DatasetManager()
        print("    Dataset manager ready")
        
        # Settings
        self.timeout_sec = float(timeout_sec)
        self.similarity_threshold = float(similarity_threshold)
        self.plate_confidence_thresh = float(plate_confidence_thresh)
        
        # State
        self.start_time = None
        self.last_embedding_time = 0
        self.checkout_plate = None
        self.checkout_face_embedding = None
        self.result = None
        self.last_face_seen_time = 0
        self.mqtt_response = None
        self.mqtt_reason = None 
        register_response_callback(self.on_mqtt_response)
        self.lot_name = "Bãi Xe Đại Học FPT"
        
        print("\n All modules initialized successfully!")
        print("="*70)

        self.verify_plate_text = None
        self.verify_start_time = None
        self.verify_wait_sec = 2.0

    def update_config(self, config):
        if config.get("lot_name") != self.lot_name:
            return

        if "similarity_threshold" in config:
            self.similarity_threshold = max(
                0.0, min(1.0, float(config["similarity_threshold"]))
            )

        if "plate_confidence_thresh" in config:
            self.plate_confidence_thresh = max(
                0.0, min(1.0, float(config["plate_confidence_thresh"]))
            )

        print(f"[CHECKOUT CONFIG UPDATED] sim={self.similarity_threshold}")

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
        print("CHECK-OUT PROCESS STARTED")
        print("="*70)
        print(f"\n⏱TIME LIMIT: {self.timeout_sec} seconds")
        print("Scanning face and license plate...")
        print("━" * 70)
        
        current_similarity_threshold = self.similarity_threshold
        self.start_time = time.time()
        # checkout_success = False
        VERIFY_COOLDOWN = 2
        last_verify_time = 0
        
        while True:
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            remaining = self.timeout_sec - elapsed
            
            # Check timeout
            if elapsed >= self.timeout_sec:
                print("TIMEOUT → SEND MQTT")

                try:
                    send_checkout(
                        plate_number="UNKNOWN",
                        similarity=0,
                        face_img=None,
                        plate_img=None,
                        status="fail",
                        reason="timeout",
                        lot_name=self.lot_name,
                        confidence_score=0,
                        processing_time_ms=int((time.time() - self.start_time) * 1000)
                    )
                except Exception as e:
                    print(f"MQTT timeout error: {e}")

                self.result = {
                    'success': False,
                    'message': 'TIMEOUT',
                    'plate': None,
                    'similarity': None,
                    'duration_sec': elapsed,
                    'reason': 'timeout'
                }

                break  
            
            run_ai = (self.frame_count % self.frame_skip == 0)
            
            face_frame = self.face_cam.get_frame()
            plate_frame = self.plate_cam.get_frame()

            if face_frame is None or plate_frame is None:
                continue

            # Detect face
            face_detected = False
            face_image = None
            face_embedding = None

            if not hasattr(self, "last_face_detected"):
                self.last_face_detected = False
                self.last_face_image = None

            face_detected = self.last_face_detected
            face_image = self.last_face_image
            face_embedding = self.checkout_face_embedding

            if run_ai:
                try:
                    faces, boxes = self.face_detector.extract_all_faces(face_frame)

                    if len(faces) > 0:
                        face_detected = True
                        face_image = faces[0]
                        self.last_face_seen_time = time.time()

                        if time.time() - self.last_embedding_time > 1:
                            face_embedding = self.face_recognizer.get_embedding(face_image)
                            self.last_embedding_time = time.time()
                            self.checkout_face_embedding = face_embedding

                        # SAVE STATE
                        self.last_face_detected = True
                        self.last_face_image = face_image

                    else:
                        # mất mặt quá 1 giây thì reset
                        if time.time() - self.last_face_seen_time > 1:
                            self.last_face_detected = False
                            self.last_face_image = None
                            self.checkout_face_embedding = None

                            face_detected = False
                            face_image = None
                            face_embedding = None

                except:
                    pass
                
            # Detect plate
            plate_detected = False
            plate_text = None
            plate_confidence = 0.0
            plate_bbox = None
            plate_image = None
            if not hasattr(self, "last_plate_detected"):
                self.last_plate_detected = False
                self.last_plate_text = None
                self.last_plate_image = None

            plate_detected = self.last_plate_detected
            plate_text = self.last_plate_text
            plate_image = self.last_plate_image

            try:
                if self.frame_count % 2 == 0:
                    detected_plates = self.plate_detector.detect(
                        plate_frame,
                        conf_threshold=self.plate_confidence_thresh
                    )
                else:
                    detected_plates = []

                if len(detected_plates) > 0:
                    best = detected_plates[0]

                    plate_confidence = float(best.get('confidence', 0.0))

                    raw_plate_text = best.get('text')
                    clean_plate = normalize_plate(raw_plate_text)

                    if clean_plate:
                        plate_detected = True
                        plate_text = clean_plate

                        plate_bbox = best.get('bbox')
                        x1, y1, x2, y2 = plate_bbox
                        plate_image = plate_frame[y1:y2, x1:x2]

                        # SAVE STATE
                        self.last_plate_detected = True
                        self.last_plate_text = plate_text
                        self.last_plate_image = plate_image

            except:
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
                if not plate_text or len(plate_text) < 8:
                    plate_detected = False
                
                if self.verify_plate_text != plate_text:
                    self.verify_plate_text = plate_text
                    self.verify_start_time = current_time
                    print(f"Plate detected ({plate_text}) — waiting {self.verify_wait_sec}s to stabilize...")

                else:
                    elapsed_verify = current_time - self.verify_start_time

                    if elapsed_verify >= self.verify_wait_sec and self.result is None and current_time - last_verify_time > VERIFY_COOLDOWN:

                        print(f"\n Plate & Face stable for {self.verify_wait_sec}s")
                        print("Verifying against database...")

                        match_result = self._verify_checkout(plate_text, face_embedding, current_similarity_threshold)

                        last_verify_time = current_time
                        elapsed_total = time.time() - self.start_time

                        # ===== SUCCESS =====
                        if match_result['success']:
                            print("\n MATCH SUCCESS → Sending MQTT")

                            try:
                                self.checkout_plate = plate_text

                                send_checkout(
                                    plate_number=plate_text,
                                    similarity=match_result.get('similarity'),
                                    face_img=face_image,
                                    plate_img=plate_image,
                                    status="success",
                                    reason="face_match",
                                    lot_name=self.lot_name,
                                    confidence_score=plate_confidence,
                                    processing_time_ms=int((time.time() - self.start_time) * 1000)
                                )

                                print("MQTT SENT → WAITING FOR BE RESPONSE")

                                self.mqtt_response = None
                                self.mqtt_reason = None

                                wait_start = time.time()
                                timeout = 5

                                while self.mqtt_response is None and time.time() - wait_start < timeout:
                                    time.sleep(0.1)

                                print(f"FAIL RESPONSE FROM BE: {self.mqtt_response} - {self.mqtt_reason}")

                                if self.mqtt_response == "OPEN":
                                    print("BE ALLOW → OPEN GATE")
                                    success_flag = True

                                elif self.mqtt_response == "DENY":
                                    print(f"BE DENY → {self.mqtt_reason}")
                                    success_flag = False

                                else:
                                    print("NO RESPONSE FROM BE → AUTO DENY")
                                    success_flag = False

                                self.result = {
                                    'success': success_flag,
                                    'message': 'CHECKOUT SUCCESS' if success_flag else 'CHECKOUT FAILED',
                                    'plate': plate_text,
                                    'similarity': match_result.get('similarity'),
                                    'duration_sec': elapsed_total,
                                    'reason': self.mqtt_reason if self.mqtt_reason else "be_timeout"
                                }

                                break

                            except Exception as e:
                                print(f"MQTT send failed: {e}")


                        # ===== FAIL (QUAN TRỌNG) =====
                        else:
                            print(f"Verification failed ({match_result['reason']})")

                            try:
                                self.checkout_plate = plate_text

                                send_checkout(
                                    plate_number=plate_text,
                                    similarity=match_result.get('similarity'),
                                    face_img=face_image,
                                    plate_img=plate_image,
                                    status="fail",
                                    reason=match_result.get('reason', 'unknown'),
                                    lot_name=self.lot_name,
                                    confidence_score=plate_confidence,
                                    processing_time_ms=int((time.time() - self.start_time) * 1000)
                                )

                                print("MQTT SENT (FAIL)")

                            except Exception as e:
                                print(f"MQTT fail send error: {e}")

                            self.result = {
                                'success': False,
                                'message': 'CHECKOUT FAILED',
                                'plate': plate_text,
                                'similarity': match_result.get('similarity'),
                                'duration_sec': elapsed_total,
                                'reason': match_result.get('reason')
                            }

                            break
            
            # Quit on 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nCancelled by user")
                self.result = {
                    'success': False,
                    'message': 'CANCELLED',
                    'plate': None,
                    'similarity': None,
                    'duration_sec': elapsed,
                    'reason': 'user_cancel'
                }
                break
        
        self.cleanup()
        self._display_result()
        
        return self.result
    
    def _verify_checkout(self, plate_text, checkout_face_embedding, threshold):
        # ===== LẤY ẢNH THEO BIỂN SỐ =====
        plate_dir = Path(self.dataset_manager.face_images_dir) / plate_text

        if not plate_dir.exists():
            return {
                'success': False,
                'reason': 'plate_not_found',
                'message': 'Khong tim thay bien so',
                'similarity': 0.0
            }

        stored_vectors = []

        for img_path in plate_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            faces, _ = self.face_detector.extract_all_faces(img)
            if len(faces) == 0:
                continue

            emb = self.face_recognizer.get_embedding(faces[0])
            stored_vectors.append(emb)

        if len(stored_vectors) == 0:
            return {
                'success': False,
                'reason': 'no_face_data',
                'message': 'Khong co du lieu khuon mat',
                'similarity': 0.0
            }

        # ===== SO SANH =====
        max_similarity = 0.0

        for stored_vector in stored_vectors:
            v1 = checkout_face_embedding / (np.linalg.norm(checkout_face_embedding) + 1e-8)
            v2 = stored_vector / (np.linalg.norm(stored_vector) + 1e-8)

            similarity = float(np.dot(v1, v2))

            if similarity > max_similarity:
                max_similarity = similarity

        if max_similarity < threshold:
            return {
                'success': False,
                'reason': 'face_not_match',
                'message': 'Khong khop khuon mat',
                'similarity': max_similarity
            }
        else:
            return {
                'success': True,
                'reason': 'face_match',
                'message': 'Khuon mat hop le',
                'similarity': max_similarity
            }
    
    def _display_result(self):
        """Hiển thị kết quả check-out"""
        if self.result is None:
            return
        
        print("\n" + "="*70)
        print(" CHECK-OUT RESULT")
        print("="*70)
        
        success = self.result['success']
        message = self.result['message']
        plate = self.result['plate']
        similarity = self.result['similarity']
        duration = self.result['duration_sec']
        reason = self.result.get('reason', 'unknown')
        
        print(f"\n{message}")
        print(f"\n Details:")
        print(f"   Plate: {plate if plate else 'N/A'}")
        print(f"   Similarity: {f'{similarity:.1%}' if similarity is not None else 'N/A'}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Reason: {reason}")
        
        print("\n" + "="*70)
    
    def cleanup(self):
        cv2.destroyAllWindows()

    def on_mqtt_response(self, payload):
        try:
            data = json.loads(payload)

            if data.get("event") != "checkout_result":
                return 

            plate = data.get("plateNumber")
            status = data.get("status") 
            reason = data.get("reason")

            print(f"[MQTT RESPONSE] plate={plate}, status={status}, reason={reason}")

            if not plate:
                return

            if self.checkout_plate and plate != self.checkout_plate:
                print("Ignore response (not current plate)")
                return

            self.mqtt_response = status
            self.mqtt_reason = reason   # 👈 QUAN TRỌNG

        except Exception as e:
            print(f"[MQTT ERROR] {e}")
def main():
    try:
        checkout = CheckOutCapture(
            face_cam_url="http://192.168.137.40:8081/video",
            plate_cam_url="http://192.168.137.227:8081/video",
            timeout_sec=60,
            similarity_threshold=0.70,
            plate_confidence_thresh=0.80
        )
        register_config_callback(checkout.update_config)
        result = checkout.start_checkout()
        
        return result
    except Exception as e:
        print(f"\nError: {e}")
        return "DENY"
    
if __name__ == "__main__":
    print("\n PARKING CHECK-OUT SYSTEM")
    print("=" * 70)
    print("\n PROCESS:")
    print("   1. Scan face and license plate")
    print("   2. Find matching record in database")
    print("   3. Compare face similarity (>= 70%)")
    print("   4. Show result (success/failure)")
    print("\n TIME LIMIT: 30 seconds")
    print(" Press 'q' to cancel\n")
        
    result = main()

    print(json.dumps(result))

    sys.exit(0)