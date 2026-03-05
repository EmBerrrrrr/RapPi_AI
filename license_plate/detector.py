"""
Vietnamese License Plate Detection & Recognition Module
"""

import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics not installed. Run: pip install ultralytics")

# Character mapping for OCR
CHAR_MAP = {
    0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",
    10:"A",11:"B",12:"C",13:"D",14:"E",15:"F",16:"G",17:"H",
    18:"I",19:"J",20:"K",21:"L",22:"M",23:"N",24:"O",25:"P",
    26:"Q",27:"R",28:"S",29:"T",30:"U",31:"V",32:"W",33:"X",
    34:"Y",35:"Z"
}


class LicensePlateDetector:
    """Detect and read Vietnamese license plates"""
    
    def __init__(self, models_dir="license_plate/models"):
        """
        Initialize detector
        
        Args:
            models_dir: Directory containing models
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed")
        
        models_path = Path(models_dir)
        lp_model = models_path / "LP_best.pt"
        letters_model = models_path / "Letters_detection.pt"
        
        if not lp_model.exists():
            raise FileNotFoundError(f"License plate model not found: {lp_model}")
        if not letters_model.exists():
            raise FileNotFoundError(f"Letters model not found: {letters_model}")
        
        print("🔄 Loading License Plate Detection models...")
        self.lp_detector = YOLO(str(lp_model))
        self.ocr_model = YOLO(str(letters_model))
        print("✅ License Plate Detection ready")
    
    def _deskew(self, image, cc, ct):
        """Rotate image if needed"""
        if cc == 1:
            image = cv2.rotate(image, cv2.ROTATE_180)
        if ct == 1:
            image = cv2.flip(image, 1)
        return image
    
    def _ocr_read(self, crop_img):
        """
        Read characters from license plate
        
        Args:
            crop_img: Cropped license plate image
            
        Returns:
            str: Detected license plate or None
        """
        try:
            result = self.ocr_model(crop_img, verbose=False)[0]
            boxes = result.boxes
            
            if len(boxes) == 0:
                return None
            
            # Get coordinates and class
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            
            # Sort by coordinates
            centers = [(xyxy[i], cls[i]) for i in range(len(xyxy))]
            
            if len(centers) == 0:
                return None
            
            # Split lines (VN plates can have 1 or 2 lines)
            y_coords = [c[0][1] for c in centers]
            y_mean = np.mean(y_coords)
            
            line1 = []
            line2 = []
            
            for box, cls_id in centers:
                x1, y1, x2, y2 = box
                center_y = (y1 + y2) / 2
                center_x = (x1 + x2) / 2
                
                if center_y < y_mean:
                    line1.append((center_x, cls_id))
                else:
                    line2.append((center_x, cls_id))
            
            # Sort left to right
            line1.sort(key=lambda x: x[0])
            line2.sort(key=lambda x: x[0])
            
            # Create text
            text1 = "".join([CHAR_MAP.get(cls_id, "?") for _, cls_id in line1])
            text2 = "".join([CHAR_MAP.get(cls_id, "?") for _, cls_id in line2])
            
            # Format license plate
            if text1 and text2:
                return f"{text1}-{text2}"
            elif text1:
                return text1
            elif text2:
                return text2
                
        except Exception as e:
            print(f"OCR error: {e}")
        
        return None
    
    def detect(self, frame, conf_threshold=0.5):
        """
        Detect license plate in frame
        
        Args:
            frame: Frame from camera
            conf_threshold: Confidence threshold
            
        Returns:
            List of detected license plates
        """
        results = []
        
        try:
            # Detect license plates
            detections = self.lp_detector(frame, conf=conf_threshold, verbose=False)
            
            for result in detections:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Crop license plate
                    lp_crop = frame[y1:y2, x1:x2]
                    
                    if lp_crop.size > 0:
                        # Try OCR with multiple orientations
                        best_text = None
                        
                        for cc in [0, 1]:
                            for ct in [0, 1]:
                                deskewed = self._deskew(lp_crop.copy(), cc, ct)
                                text = self._ocr_read(deskewed)
                                
                                if text and len(text) >= 6:
                                    best_text = text
                                    break
                            if best_text:
                                break
                        
                        results.append({
                            'bbox': (x1, y1, x2, y2),
                            'text': best_text if best_text else "Unknown",
                            'confidence': conf
                        })
            
        except Exception as e:
            print(f"Detection error: {e}")
        
        return results
