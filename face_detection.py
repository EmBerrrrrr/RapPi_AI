"""
Face detection module using MTCNN (PyTorch)
"""

from facenet_pytorch import MTCNN
import cv2
import numpy as np
import torch
from config import FACE_SIZE, MIN_FACE_SIZE


class FaceDetector:
    """Detect and crop faces using MTCNN (PyTorch)"""
    
    def __init__(self):
        """Initialize MTCNN detector"""
        print("🔄 Initializing MTCNN detector (PyTorch)...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            min_face_size=MIN_FACE_SIZE,
            device=device,
            keep_all=True
        )
        print(f"✅ MTCNN detector ready - Device: {device}")
    
    def detect_faces(self, image):
        """
        Detect all faces in image
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            List of detected faces (bounding boxes)
        """
        # Chuyển BGR sang RGB (MTCNN yêu cầu RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces - returns boxes, probs
        boxes, probs = self.detector.detect(rgb_image)
        
        if boxes is None:
            return []
        
        # Convert to compatible format
        results = []
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = [int(b) for b in box]
            results.append({
                'box': [x1, y1, x2-x1, y2-y1],  # [x, y, w, h]
                'confidence': float(prob)
            })
        
        return results
    
    def extract_face(self, image, required_size=FACE_SIZE):
        """
        Extract first face from image
        
        Args:
            image: BGR image from OpenCV
            required_size: Output size (width, height)
            
        Returns:
            Resized face array, or None if not found
        """
        results = self.detect_faces(image)
        
        if len(results) == 0:
            return None
        
        # Get first face
        x, y, w, h = results[0]['box']
        
        # Ensure coordinates are not negative
        x, y = abs(x), abs(y)
        
        # Crop face
        face = image[y:y+h, x:x+w]
        
        # Resize to standard size for FaceNet
        face = cv2.resize(face, required_size)
        
        return face
    
    def extract_all_faces(self, image, required_size=FACE_SIZE):
        """
        Extract ALL faces from image
        
        Args:
            image: BGR image from OpenCV
            required_size: Output size
            
        Returns:
            List of face arrays and bounding boxes
        """
        results = self.detect_faces(image)
        
        faces = []
        boxes = []
        
        for result in results:
            x, y, w, h = result['box']
            x, y = abs(x), abs(y)
            
            # Crop and resize
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, required_size)
            
            faces.append(face)
            boxes.append((x, y, w, h))
        
        return faces, boxes
    
    def draw_faces(self, image, show_confidence=True):
        """
        Draw bounding boxes on detected faces
        
        Args:
            image: BGR image from OpenCV
            show_confidence: Whether to show confidence score
            
        Returns:
            Image with drawn bounding boxes
        """
        results = self.detect_faces(image)
        
        output_image = image.copy()
        
        for result in results:
            x, y, w, h = result['box']
            confidence = result['confidence']
            
            # Draw bounding box
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw confidence score
            if show_confidence:
                text = f"{confidence:.2f}"
                cv2.putText(output_image, text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output_image


# Test function
if __name__ == "__main__":
    print("🧪 Testing MTCNN Face Detection...")
    
    detector = FaceDetector()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    print("📸 Press 'q' to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw faces
        output = detector.draw_faces(frame)
        
        cv2.imshow("MTCNN Face Detection Test", output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test completed")
