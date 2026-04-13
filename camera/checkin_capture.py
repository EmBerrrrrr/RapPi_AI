"""
Dual Camera Capture - Quét khuôn mặt và biển số cùng lúc
Chỉ lưu dataset khi phát hiện được cả 2 với nhau
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
from mqtt_client import register_config_callback

from face_recognition.face_detection import FaceDetector
from face_recognition.face_recognition import FaceRecognizer
from license_plate.detector import LicensePlateDetector
from dataset_manager import DatasetManager
from mqtt_client import send_checkin
from datetime import datetime, timezone
from plate_utils import normalize_plate
from ip_camera import IPCamera

class CheckInCapture:
    """
    Mở camera để quét khuôn mặt + biển số cùng lúc
    Chỉ lưu dataset khi phát hiện cả 2
    """
    
    def __init__(self,frame_skip = 2, last_embedding_time = 0, face_cam_url=None, plate_cam_url=None, save_interval=60, face_blur_thresh=50.0, plate_confidence_thresh=0.8, min_face_size=240, face_quality_percent_thresh=0.8, auto_stop_after_save=False, last_plate_logged=None):
        """
        Khởi tạo camera capture
        
        Args:
            save_interval: Số frames giữa các lần lưu (tránh spam)
        """
        print("\n" + "="*70)
        print("DUAL CAMERA CAPTURE - INITIALIZATION")
        
        print("="*70)
        
        # Initialize camera
        print("\nInitializing camera...")
        # Camera quét mặt (Laptop)
        self.face_cam = IPCamera(face_cam_url)
        self.plate_cam = IPCamera(plate_cam_url)
        
        self.frame_skip = frame_skip
        self.last_embedding_time = last_embedding_time
        
        print(f"Camera opened (URL mode)")
        
        # Initialize AI modules
        print("\n Initializing AI modules...")
        self.face_detector = FaceDetector()
        print("   Face detector ready")
        
        self.face_recognizer = FaceRecognizer()
        print("   Face recognizer ready")
        
        self.plate_detector = LicensePlateDetector()
        print("   License plate detector ready")
        
        # Initialize dataset manager
        self.dataset_manager = DatasetManager()
        print("   Dataset manager ready")
        
        # Settings
        self.save_interval = save_interval  # Seconds between saves
        # Quality thresholds
        self.face_blur_thresh = float(face_blur_thresh)  # variance of Laplacian (absolute)
        self.plate_confidence_thresh = float(plate_confidence_thresh)  # value between 0-1

        self.min_face_size = int(min_face_size)
        # Alternative percent-based face quality check (0-1). If set, use this instead of absolute blur threshold.
        self.face_quality_percent_thresh = None if face_quality_percent_thresh is None else float(face_quality_percent_thresh)
        # If True, stop capture after a successful save
        self.auto_stop_after_save = bool(auto_stop_after_save)
        self.frame_count = 0
        self.last_save_time = 0
        self.ready_start_time = None     # thời điểm bắt đầu đủ điều kiện
        self.save_delay_sec = 5.0       # CHỜ 5 GIÂY RỒI MỚI LƯU
        self.plate_lock_text = None
        self.plate_lock_start_time = None
        self.plate_lock_sec = 2.0
        self.last_face_embedding = None
        # Statistics
        self.face_count = 0
        self.plate_count = 0
        self.saved_count = 0
        self.last_saved_plate = None
        self.lot_name = "Bãi Xe Đại Học FPT"
        
        print("\nAll modules initialized successfully!")
        print("="*70)
    
    def update_config(self, config):
        if config.get("lot_name") != self.lot_name:
            return

        if "plate_confidence_thresh" in config:
            self.plate_confidence_thresh = max(
                0.0, min(1.0, float(config["plate_confidence_thresh"]))
            )

        print(f"[CHECKIN CONFIG UPDATED] plate={self.plate_confidence_thresh}")
    
    def detect_and_capture(self):
        """
        Chạy camera loop để phát hiện và lưu dataset
        
        Controls:
            's' - Manual save (nếu phát hiện cả 2)
            'r' - Show report
            'q' - Quit
        """
        print("\n Starting camera capture...")
        print("━" * 70)
        print(" Controls:")
        print("   's' - Manual save (if both detected)")
        print("   'r' - Show report")
        print("   'd' - Toggle debug mode")
        print("   'q' - Quit")
        print("━" * 70 + "\n")
        
        debug_mode = False
        
        while True:
            face_frame = self.face_cam.get_frame()
            plate_frame = self.plate_cam.get_frame()

            if face_frame is None or plate_frame is None:
                continue

            face_display = face_frame.copy()
            plate_display = plate_frame.copy()

            self.frame_count += 1
            run_ai = (self.frame_count % self.frame_skip == 0)
            
            # Detect faces
            if not hasattr(self, "last_face_detected"):
                self.last_face_detected = False
                self.last_face_image = None

            face_detected = self.last_face_detected
            face_image = self.last_face_image
            face_embedding = self.last_face_embedding

            if run_ai:
                try:
                    faces, boxes = self.face_detector.extract_all_faces(face_frame)

                    if len(faces) > 0:
                        face_detected = True
                        face_image = faces[0]

                        if time.time() - self.last_embedding_time > 1:
                            face_embedding = self.face_recognizer.get_embedding(face_image)
                            self.last_embedding_time = time.time()
                            self.last_face_embedding = face_embedding

                        # SAVE STATE
                        self.last_face_detected = True
                        self.last_face_image = face_image
                        self.face_count += 1

                except:
                    pass
                
            if not hasattr(self, "last_plate_detected"):
                self.last_plate_detected = False
                self.last_plate_text = None
                self.last_plate_image = None
                self.last_plate_bbox = None
                self.last_plate_confidence = 0.0

            plate_detected = self.last_plate_detected
            plate_text = self.last_plate_text
            plate_image = self.last_plate_image
            plate_bbox = self.last_plate_bbox
            plate_confidence = self.last_plate_confidence
            plate_text_stable = None

            try:
                if self.frame_count % 2 == 0:
                    detected_plates = self.plate_detector.detect(plate_frame, conf_threshold=0.4)
                else:
                    detected_plates = []

                if len(detected_plates) > 0:
                    best_result = detected_plates[0]

                    raw_plate_text = best_result.get('text')
                    clean_plate = normalize_plate(raw_plate_text)

                    if clean_plate:
                        plate_detected = True
                        plate_text = clean_plate

                        plate_bbox = best_result.get('bbox')
                        plate_confidence = float(best_result.get('confidence', 0.0))

                        # ===== STABLE TEXT (GIỮ NGUYÊN) =====
                        if plate_text:
                            now = time.time()
                            if self.plate_lock_text != plate_text:
                                self.plate_lock_text = plate_text
                                self.plate_lock_start_time = now
                            else:
                                if now - self.plate_lock_start_time >= self.plate_lock_sec:
                                    plate_text_stable = self.plate_lock_text

                            if plate_text_stable:
                                plate_text = plate_text_stable

                        # ===== CROP =====
                        if plate_bbox and len(plate_bbox) == 4:
                            x1, y1, x2, y2 = plate_bbox
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(plate_frame.shape[1], x2), min(plate_frame.shape[0], y2)

                            if x2 > x1 and y2 > y1:
                                plate_image = plate_frame[y1:y2, x1:x2]

                        # ===== SAVE STATE (QUAN TRỌNG NHẤT) =====
                        self.last_plate_detected = True
                        self.last_plate_text = plate_text
                        self.last_plate_image = plate_image
                        self.last_plate_bbox = plate_bbox
                        self.last_plate_confidence = plate_confidence

                        # COUNT
                        if plate_text and plate_text != "Unknown" and plate_text != self.last_saved_plate:
                            self.plate_count += 1

                else:
                    pass

            except Exception as e:
                print(f"Plate detection error: {e}")
            
            # Draw on frame
            face_display = face_frame.copy()
            plate_display = plate_frame.copy()

            
            # Draw face boxes
            try:
                face_display = self.face_detector.draw_faces(face_display)
            except:
                pass
            
            # Draw plate detection
            if plate_detected and plate_text and plate_bbox:
                try:
                    x1, y1, x2, y2 = plate_bbox
                    cv2.rectangle(plate_display, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(plate_display, plate_text, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                except:
                    pass
            
            # Status text
            status_text = f"Faces: {self.face_count} | Plates: {self.plate_count} | Saved: {self.saved_count}"
            cv2.putText(face_display, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Debug info (shows blur metric and plate confidence)
            if debug_mode:
                blur_val = None
                if face_image is not None:
                    try:
                        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                        blur_val = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                    except:
                        blur_val = 0

                # plate_confidence may not be defined if detection failed
                try:
                    pc = plate_confidence
                except NameError:
                    pc = 0.0

                blur_str = f"{blur_val:.1f}" if isinstance(blur_val, (int, float)) else "N/A"

                debug_info = f"F:{face_detected} P:{plate_detected} Txt:{plate_text} Conf:{pc:.2f} Blur:{blur_str}"

                cv2.putText(face_display, debug_info, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Detection status
            detection_status = " NO DETECTION"
            if face_detected and plate_detected:
                detection_status = " BOTH DETECTED - READY TO SAVE"
                cv2.putText(face_display, detection_status, (10, 470),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif face_detected:
                detection_status = " FACE DETECTED (waiting for plate)"
                cv2.putText(face_display, detection_status, (10, 470),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif plate_detected:
                detection_status = " PLATE DETECTED (waiting for face)"
                cv2.putText(plate_display, detection_status, (10, 470),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display

            face_display_small = cv2.resize(face_display, (640, 480))
            plate_display_small = cv2.resize(plate_display, (480, 270))  # 👈 thu nhỏ cam điện thoại

            cv2.imshow("FACE CAMERA", face_display_small)
            cv2.imshow("PLATE CAMERA", plate_display_small)


            # Check quality before saving
            current_time = time.time()
            quality_ok = False
            reason = None

            if face_detected and plate_detected and face_image is not None and plate_image is not None and face_embedding is not None and plate_text is not None:
                # Plate confidence check
                try:
                    pc = float(plate_confidence)
                except Exception:
                    pc = 0.0

                if pc < self.plate_confidence_thresh:
                    reason = f"Low plate confidence: {pc:.2f} (<{self.plate_confidence_thresh})"
                else:
                    # Face blur check
                    try:
                        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                        blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                        if blur_score < self.face_blur_thresh:
                            reason = f"Face too blurry: {blur_score:.1f}"
                            #reason = f"Face too blurry: {blur_score:.1f} (<{self.face_blur_thresh})"
                        #elif face_image.shape[0] < self.min_face_size or face_image.shape[1] < self.min_face_size:
                        #   reason = f"Face too small: {face_image.shape} (<{self.min_face_size}px)"
                        else:
                            quality_ok = True
                    except Exception as e:
                        reason = f"Face quality check error: {e}"
                        
            ready_to_save = False
                        
            if quality_ok:
                if self.ready_start_time is None:
                    self.ready_start_time = current_time
                elif current_time - self.ready_start_time >= self.save_delay_sec:
                    ready_to_save = True
            else:
                self.ready_start_time = None
                ready_to_save = False
                        
                        
            # Detect_and_capture--------------------------------------------------
            if ready_to_save and plate_text_stable and plate_text_stable != self.last_saved_plate:
                print("SAVING AFTER 5s STABLE DETECTION")

                success = self._save_face_and_plate(
                    face_image,
                    face_embedding,
                    plate_text_stable,
                    plate_image,
                    plate_confidence
                )

                if success:
                    self.last_saved_plate = plate_text_stable

                    print("🛑 Auto stop camera after save")
                    self.cleanup()
                    return True   # SUCCESS
                else:
                    print("Save failed")
                    return False
            #--------------------------------------------------------------
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n🛑 Quitting...")
                break
            elif key == ord('r'):
                self._show_report()
            elif key == ord('d'):
                debug_mode = not debug_mode
                status = "ON" if debug_mode else "OFF"
                print(f"🐛 Debug mode: {status}")
            elif key == ord('s'):
                # Manual save
                if face_detected and plate_detected and face_image is not None and plate_image is not None:
                    self._save_face_and_plate(face_image, face_embedding, plate_text, plate_image)
                    self.last_saved_plate = plate_text_stable
                else:
                    print("⚠️  Cannot save: need both face and plate detected!")
        
        self.cleanup()
        return False
    
    def _save_face_and_plate(self, face_image, face_embedding, plate_text, plate_image, plate_confidence):
        try:
            clean_plate = plate_text.strip().upper().replace(" ", "_") if plate_text else "UNKNOWN"

            print(f"\nSAVING DATA FOR: {clean_plate}")

            face_saved = self.dataset_manager.save_face_vector(
                name=clean_plate,
                face_image=face_image,
                embedding_vector=face_embedding,
                metadata={
                    'source': 'dual_camera',
                    'plate': plate_text,
                    'timestamp': datetime.now().isoformat()
                }
            )

            if not face_saved:
                print("Face save failed")
                send_checkin(
                    plate_number=clean_plate,
                    face_img=None,
                    plate_img=None,
                    status="fail",
                    reason="face_save_failed",
                    lot_name=self.lot_name,
                    confidence_score=plate_confidence,
                    processing_time_ms=int(time.time() * 1000)
                )
                return False

            print("Face saved")

            plate_saved = self.dataset_manager.save_license_plate(
                plate_text=plate_text,
                plate_image=plate_image,
                metadata={
                    'source': 'dual_camera',
                    'timestamp': datetime.now().isoformat()
                }
            )

            if not plate_saved:
                print("Plate save failed")
                
                
                send_checkin(
                    plate_number=clean_plate,
                    face_img=None,
                    plate_img=None,
                    status="fail",
                    reason="plate_save_failed",
                    lot_name=self.lot_name,
                    confidence_score=plate_confidence,
                    processing_time_ms=int(time.time() * 1000)
                )
                return False

            print("Plate saved")

            self.dataset_manager.record_checkin(
                plate_text=plate_text,
                face_name=clean_plate,
                metadata={
                    'source': 'dual_camera',
                    'timestamp': datetime.now().isoformat()
                }
            )

            self.saved_count += 1
            
            
            send_checkin(
                plate_number=clean_plate,
                face_img=face_image,
                plate_img=plate_image,
                status="success",
                reason="ok",
                lot_name=self.lot_name,
                confidence_score=plate_confidence,
                processing_time_ms=int(time.time() * 1000)
            )

            print("CHECK-IN SUCCESS SENT")
            return True

        except Exception as e:
            print(f"Exception: {e}")
            
            
            send_checkin(
                plate_number="UNKNOWN",
                face_img=None,
                plate_img=None,
                status="fail",
                reason="exception",
                lot_name=self.lot_name,
                confidence_score=plate_confidence,
                processing_time_ms=int(time.time() * 1000)
            )

            return False
    
    def _show_report(self):
        """Hiển thị báo cáo thống kê"""
        print("\n" + "="*70)
        print(" STATISTICS REPORT")
        print("="*70)
        
        print(f"\n Camera Statistics:")
        print(f"   Total frames: {self.frame_count}")
        print(f"   Faces detected: {self.face_count}")
        print(f"   Plates detected: {self.plate_count}")
        print(f"   Saved pairs: {self.saved_count}")
        
        # Get dataset stats
        summary = self.dataset_manager.get_summary()
        
        print(f"\n Face Database:")
        print(f"   Total persons: {summary['faces']['total_persons']}")
        print(f"   Total vectors: {summary['faces']['total_vectors']}")
        
        print(f"\n License Plate Database:")
        print(f"   Total plates: {summary['license_plates']['total_unique_plates']}")
        print(f"   Total images: {summary['license_plates']['total_images']}")
        
        print("\n" + "="*70 + "\n")
    
    def cleanup(self):
        """Dọn dẹp resources"""
        print("\n Cleaning up...")
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print(" FINAL REPORT")
        print("="*70)
        print(f"\nTotal frames processed: {self.frame_count}")
        print(f"Faces detected: {self.face_count}")
        print(f"Plates detected: {self.plate_count}")
        print(f"Face-Plate pairs saved: {self.saved_count}")
        
        # Final dataset stats
        summary = self.dataset_manager.get_summary()
        print(f"\from django.utils.translation import ungettextFinal Dataset Status:")
        print(f"    Persons (by plate): {summary['faces']['total_persons']}")
        print(f"   Face vectors: {summary['faces']['total_vectors']}")
        print(f"    Unique plates: {summary['license_plates']['total_unique_plates']}")
        print(f"    Plate images: {summary['license_plates']['total_images']}")
        
        print(f"\nSaved to:")
        print(f"   Faces: {summary['directories']['face_images']}")
        print(f"   Plates: {summary['directories']['lp_images']}")
        print("\n" + "="*70)


def main():
    try:
        capture = CheckInCapture(
            face_cam_url="http://192.168.137.129:8081/video",
            plate_cam_url="http://192.168.137.227:8081/video",
            save_interval=60
        )
        register_config_callback(capture.update_config)
        result = capture.detect_and_capture()
        if result:
            return "OPEN"
        else:
            return "DENY"
    except Exception as e:
        print(f"\n Error: {e}")
        return "DENY"
    
if __name__ == "__main__":
    print("\n DUAL CAMERA CAPTURE WITH DATASET SAVING")
    print("=" * 70)

    print("\n    REQUIREMENTS:")
    print("   1. Camera connected")
    print("   2. All AI models loaded")
    print("   3. Position yourself in front of camera with vehicle")

    print("\nBEHAVIOR:")
    print("   • Detects faces continuously")
    print("   • Detects license plates continuously")
    print("   • Saves ONLY when BOTH are detected together")
    print("   • Each plate is saved once (no spam)")
    print("   • Face saved with same name as plate")
    print("\n")

    result = main()
    print(json.dumps({
            "success": True if result == "OPEN" else False
        }))