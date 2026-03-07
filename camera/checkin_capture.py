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

from face_recognition.face_detection import FaceDetector
from face_recognition.face_recognition import FaceRecognizer
from license_plate.detector import LicensePlateDetector
from dataset_manager import DatasetManager
from mqtt_client import send_checkin
from datetime import datetime, timezone


class CheckInCapture:
    """
    Mở camera để quét khuôn mặt + biển số cùng lúc
    Chỉ lưu dataset khi phát hiện cả 2
    """
    
    def __init__(self, face_cam_id=0, plate_cam_id=1, save_interval=60, face_blur_thresh=100.0, plate_confidence_thresh=0.8, min_face_size=240, face_quality_percent_thresh=0.8, auto_stop_after_save=False):
        """
        Khởi tạo camera capture
        
        Args:
            save_interval: Số frames giữa các lần lưu (tránh spam)
        """
        print("\n" + "="*70)
        print("🎬 DUAL CAMERA CAPTURE - INITIALIZATION")
        print("="*70)
        
        # Initialize camera
        print("\n📸 Initializing camera...")
        # Camera quét mặt (Laptop)
        self.face_cap = cv2.VideoCapture(face_cam_id)
        if not self.face_cap.isOpened():
            raise RuntimeError("❌ Cannot open FACE camera")

        # Camera quét biển số (Iriun Webcam – điện thoại)
        self.plate_cap = cv2.VideoCapture(plate_cam_id)
        if not self.plate_cap.isOpened():
            raise RuntimeError("❌ Cannot open PLATE camera")
        
        if not self.face_cap.isOpened():
            raise RuntimeError(f"❌ Cannot open face camera")
             
        # Set face camera resolution
        self.face_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.face_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.face_cap.set(cv2.CAP_PROP_FPS, 30)
        # Set plate camera resolution
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
        self.save_delay_sec = 10.0       # CHỜ 10 GIÂY RỒI MỚI LƯU
        self.plate_lock_text = None
        self.plate_lock_start_time = None
        self.plate_lock_sec = 2.0
        # Statistics
        self.face_count = 0
        self.plate_count = 0
        self.saved_count = 0
        self.last_saved_plate = None
        
        print("\n✅ All modules initialized successfully!")
        print("="*70)
    
    def detect_and_capture(self):
        """
        Chạy camera loop để phát hiện và lưu dataset
        
        Controls:
            's' - Manual save (nếu phát hiện cả 2)
            'r' - Show report
            'q' - Quit
        """
        print("\n📹 Starting camera capture...")
        print("━" * 70)
        print("🎮 Controls:")
        print("   's' - Manual save (if both detected)")
        print("   'r' - Show report")
        print("   'd' - Toggle debug mode")
        print("   'q' - Quit")
        print("━" * 70 + "\n")
        
        debug_mode = False
        
        while True:
            ret_face, face_frame = self.face_cap.read()
            ret_plate, plate_frame = self.plate_cap.read()

            if not ret_face or face_frame is None:
                continue

            if not ret_plate or plate_frame is None:
                continue

            face_display = face_frame.copy()
            plate_display = plate_frame.copy()

            if not ret_face or not ret_plate:
                print("❌ Camera read error")
                break

            self.frame_count += 1
            
            # Detect faces
            face_detected = False
            face_image = None
            face_embedding = None
            
            try:
                faces, boxes = self.face_detector.extract_all_faces(face_frame)
                if len(faces) > 0:
                    face_detected = True
                    face_image = faces[0]  # Lấy khuôn mặt đầu tiên
                    face_embedding = self.face_recognizer.get_embedding(face_image)
                    self.face_count += 1
            except Exception as e:
                pass
            plate_detected = False
            plate_text = None
            plate_image = None
            plate_bbox = None
            plate_confidence = 0.0
            plate_text_stable = None
            
            try:
                detected_plates = self.plate_detector.detect(plate_frame, conf_threshold=0.4)
                if len(detected_plates) > 0:
                    plate_detected = True
                    best_result = detected_plates[0]
                    plate_text = best_result.get('text')
                    plate_bbox = best_result.get('bbox')
                    plate_confidence = float(best_result.get('confidence', 0.0))

                    plate_text_stable = None

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

                    # Crop plate from frame if bbox valid
                    if plate_bbox and len(plate_bbox) == 4:
                        x1, y1, x2, y2 = plate_bbox
                        # clamp coords
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(plate_frame.shape[1], x2), min(plate_frame.shape[0], y2)
                        if x2 > x1 and y2 > y1:
                            plate_image = plate_frame[y1:y2, x1:x2]

                    # Chỉ đếm nếu plate text hợp lệ và khác plate đã lưu
                    if plate_text and plate_text != "Unknown" and plate_text != self.last_saved_plate:
                        self.plate_count += 1
            except Exception as e:
                print(f"⚠️  Plate detection error: {e}")
            
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
            detection_status = "🔴 NO DETECTION"
            if face_detected and plate_detected:
                detection_status = "🟢 BOTH DETECTED - READY TO SAVE"
                cv2.putText(face_display, detection_status, (10, 470),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif face_detected:
                detection_status = "🟡 FACE DETECTED (waiting for plate)"
                cv2.putText(face_display, detection_status, (10, 470),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif plate_detected:
                detection_status = "🟡 PLATE DETECTED (waiting for face)"
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
                        
            #--------------------------------------------------------------
            if quality_ok and plate_text_stable and plate_text_stable != self.last_saved_plate:
                # Kiểm tra thời gian chờ
                print("💾 SAVING AFTER 10s STABLE DETECTION")

                self._save_face_and_plate(
                    face_image,
                    face_embedding,
                    plate_text,
                    plate_image
                )

                self.last_saved_plate = plate_text

                print("🛑 Auto stop camera after save")
                self.cleanup()
                return
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
    
    def _save_face_and_plate(self, face_image, face_embedding, plate_text, plate_image):
        """
        Lưu khuôn mặt và biển số với tên giống nhau
        
        Args:
            face_image: Face image (160x160x3)
            face_embedding: Face embedding vector (512,)
            plate_text: License plate text (e.g., "29S-12345")
            plate_image: Plate image (numpy array)
        """
        try:
            # Clean plate text
            clean_plate = plate_text.strip().upper().replace(" ", "_")
            
            print(f"\n✅ SAVING DATA FOR: {clean_plate}")
            print("   " + "-" * 60)
            
            # Save face with plate name
            print(f"   📷 Saving face image...")
            face_saved = self.dataset_manager.save_face_vector(
                name=clean_plate,  # Use plate as face name
                face_image=face_image,
                embedding_vector=face_embedding,
                metadata={
                    'source': 'dual_camera',
                    'plate': plate_text,
                    'timestamp': datetime.now().isoformat()
                }
            )

            if not face_saved:
                print("   ❌ Face vector save failed — skipping plate save to avoid inconsistency")
                print("   " + "-" * 60)
                return False

            print(f"      ✅ Face saved")

            # Save plate
            print(f"   🚗 Saving license plate image...")
            plate_saved = self.dataset_manager.save_license_plate(
                plate_text=plate_text,
                plate_image=plate_image,
                metadata={
                    'source': 'dual_camera',
                    'timestamp': datetime.now().isoformat()
                }
            )

            if not plate_saved:
                print("   ❌ Plate save failed")
                print("   " + "-" * 60)
                return False

            print(f"      ✅ Plate saved")

            # Send MQTT check-in event
            try:
                send_checkin(
                    plate_number=plate_text,
                    face_img=face_image,
                    plate_img=plate_image,
                    camera_ip="192.168.1.20",
                    lot_id="0c3b5fb8-a45b-4726-b2b3-a0c3a0ae25b8",
                    gate_id=None
                )
                print("📡 MQTT check-in sent")
            except Exception as e:
                print(f"⚠️ MQTT send failed: {e}")

            print("   " + "-" * 60)
            print(f"   🎉 Total saved: {self.saved_count}\n")

            # >>> ADDED: record check-in time
            self.dataset_manager.record_checkin(
                plate_text=plate_text,
                face_name=clean_plate,
                metadata={
                    'source': 'dual_camera',
                    'timestamp': datetime.now().isoformat()
                }
            )

            self.saved_count += 1

            print("   " + "-" * 60)
            print(f"   🎉 Total saved: {self.saved_count}\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error saving: {e}")
            import traceback
            traceback.print_exc()
    
    def _show_report(self):
        """Hiển thị báo cáo thống kê"""
        print("\n" + "="*70)
        print("📊 STATISTICS REPORT")
        print("="*70)
        
        print(f"\n📹 Camera Statistics:")
        print(f"   Total frames: {self.frame_count}")
        print(f"   Faces detected: {self.face_count}")
        print(f"   Plates detected: {self.plate_count}")
        print(f"   Saved pairs: {self.saved_count}")
        
        # Get dataset stats
        summary = self.dataset_manager.get_summary()
        
        print(f"\n👤 Face Database:")
        print(f"   Total persons: {summary['faces']['total_persons']}")
        print(f"   Total vectors: {summary['faces']['total_vectors']}")
        
        print(f"\n🚗 License Plate Database:")
        print(f"   Total plates: {summary['license_plates']['total_unique_plates']}")
        print(f"   Total images: {summary['license_plates']['total_images']}")
        
        print("\n" + "="*70 + "\n")
    
    def cleanup(self):
        """Dọn dẹp resources"""
        print("\n🧹 Cleaning up...")
        self.face_cap.release()
        self.plate_cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print("📊 FINAL REPORT")
        print("="*70)
        print(f"\n✅ Total frames processed: {self.frame_count}")
        print(f"✅ Faces detected: {self.face_count}")
        print(f"✅ Plates detected: {self.plate_count}")
        print(f"✅ Face-Plate pairs saved: {self.saved_count}")
        
        # Final dataset stats
        summary = self.dataset_manager.get_summary()
        print(f"\n📊 Final Dataset Status:")
        print(f"   👤 Persons (by plate): {summary['faces']['total_persons']}")
        print(f"   🔢 Face vectors: {summary['faces']['total_vectors']}")
        print(f"   🚗 Unique plates: {summary['license_plates']['total_unique_plates']}")
        print(f"   📷 Plate images: {summary['license_plates']['total_images']}")
        
        print(f"\n📁 Saved to:")
        print(f"   Faces: {summary['directories']['face_images']}")
        print(f"   Plates: {summary['directories']['lp_images']}")
        print("\n" + "="*70)


def main():
    """Main entry point"""
    try:
        # Create capture instance
        capture = CheckInCapture(
            face_cam_id=0,
            plate_cam_id=1,
            save_interval=60
        )

        # Start camera loop
        capture.detect_and_capture()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🚀 DUAL CAMERA CAPTURE WITH DATASET SAVING")
    print("=" * 70)
    print("\n⚠️  REQUIREMENTS:")
    print("   1. Camera connected")
    print("   2. All AI models loaded")
    print("   3. Position yourself in front of camera with vehicle")
    print("\n💡 BEHAVIOR:")
    print("   • Detects faces continuously")
    print("   • Detects license plates continuously")
    print("   • Saves ONLY when BOTH are detected together")
    print("   • Each plate is saved once (no spam)")
    print("   • Face saved with same name as plate")
    print("\n")
    
    main()
