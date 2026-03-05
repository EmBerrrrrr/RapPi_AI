"""
REST API for AI Services
Flask API that .NET Backend can call
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import AI services
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from license_plate.detector import LicensePlateDetector
from parking_service import ParkingService
from database_models import create_db_engine
import uuid

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for .NET to call

# Initialize AI models (global)
print("🚀 Initializing AI models...")
face_detector = FaceDetector()
face_recognizer = FaceRecognizer()
plate_detector = LicensePlateDetector()

# Initialize database connection
print("🔌 Connecting to database...")
try:
    engine, Session = create_db_engine(
        host=os.getenv('DB_HOST'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'postgres'),
        username=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD')
    )
    print("✅ Database connected successfully!")
except Exception as e:
    print(f"⚠️ Database connection failed: {e}")
    print("⚠️ API will run without database (limited functionality)")
    engine = None
    Session = None

print("✅ AI API Server ready!")


# Helper functions
def decode_base64_image(base64_string):
    """Decode base64 image to numpy array"""
    try:
        # Remove header if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(img_data))
        img_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


# API Endpoints

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'services': {
            'face_detection': face_detector is not None,
            'face_recognition': face_recognizer is not None,
            'plate_detection': plate_detector is not None,
            'database': engine is not None
        }
    })


@app.route('/api/detect/face', methods=['POST'])
def detect_face():
    """
    Detect faces in image
    
    Request body:
    {
        "image": "base64_encoded_image"
    }
    
    Response:
    {
        "success": true,
        "faces": [{"box": [x, y, w, h], "confidence": 0.99}],
        "count": 1
    }
    """
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'success': False, 'message': 'No image provided'}), 400
        
        # Decode image
        image = decode_base64_image(image_base64)
        if image is None:
            return jsonify({'success': False, 'message': 'Invalid image format'}), 400
        
        # Detect faces
        faces = face_detector.detect_faces(image)
        
        return jsonify({
            'success': True,
            'faces': faces,
            'count': len(faces)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/recognize/face', methods=['POST'])
def recognize_face():
    """
    Recognize face from image
    
    Request body:
    {
        "image": "base64_encoded_image"
    }
    
    Response:
    {
        "success": true,
        "name": "John Doe",
        "confidence": 0.85
    }
    """
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'success': False, 'message': 'No image provided'}), 400
        
        image = decode_base64_image(image_base64)
        if image is None:
            return jsonify({'success': False, 'message': 'Invalid image format'}), 400
        
        # Detect and recognize
        faces = face_detector.detect_faces(image)
        
        if not faces:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        # Get first face
        x, y, w, h = faces[0]['box']
        face_crop = image[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (160, 160))
        
        # Recognize
        name, confidence = face_recognizer.recognize(face_crop)
        
        return jsonify({
            'success': True,
            'name': name,
            'confidence': float(confidence),
            'box': faces[0]['box']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/detect/plate', methods=['POST'])
def detect_plate():
    """
    Detect and recognize license plate
    
    Request body:
    {
        "image": "base64_encoded_image"
    }
    
    Response:
    {
        "success": true,
        "plates": [
            {
                "text": "51A-123.45",
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.95
            }
        ]
    }
    """
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'success': False, 'message': 'No image provided'}), 400
        
        image = decode_base64_image(image_base64)
        if image is None:
            return jsonify({'success': False, 'message': 'Invalid image format'}), 400
        
        # Detect plates
        plates = plate_detector.detect(image)
        
        return jsonify({
            'success': True,
            'plates': plates,
            'count': len(plates)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/parking/checkin', methods=['POST'])
def parking_checkin():
    """
    Process parking check-in
    
    Request body:
    {
        "face_image": "base64_encoded_image",
        "plate_image": "base64_encoded_image",
        "device_id": "uuid",
        "lot_id": "uuid" (optional)
    }
    
    Response:
    {
        "success": true,
        "session_id": "uuid",
        "license_plate": "51A-123.45",
        "actor_type": "guest|registered_user|monthly_pass_user",
        "user_name": "John Doe",
        "display_message": "Welcome message...",
        "barrier_action": "OPEN"
    }
    """
    try:
        data = request.get_json()
        
        face_base64 = data.get('face_image')
        plate_base64 = data.get('plate_image')
        device_id = data.get('device_id')
        lot_id = data.get('lot_id')
        
        if not face_base64 or not plate_base64:
            return jsonify({'success': False, 'message': 'Missing images'}), 400
        
        if not device_id:
            return jsonify({'success': False, 'message': 'Missing device_id'}), 400
        
        # Decode images
        face_image = decode_base64_image(face_base64)
        plate_image = decode_base64_image(plate_base64)
        
        if face_image is None or plate_image is None:
            return jsonify({'success': False, 'message': 'Invalid image format'}), 400
        
        # Create parking service with database session
        db_session = Session()
        try:
            parking_service = ParkingService(
                db_session,
                face_detector,
                face_recognizer,
                plate_detector
            )
            
            # Process check-in
            result = parking_service.process_check_in(
                face_image,
                plate_image,
                uuid.UUID(device_id),
                uuid.UUID(lot_id) if lot_id else None
            )
            
            db_session.commit()
            return jsonify(result)
            
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/parking/checkout', methods=['POST'])
def parking_checkout():
    """
    Process parking check-out
    
    Request body:
    {
        "face_image": "base64_encoded_image",
        "plate_image": "base64_encoded_image",
        "device_id": "uuid"
    }
    
    Response:
    {
        "success": true,
        "session_id": "uuid",
        "license_plate": "51A-123.45",
        "parking_fee": 15000.00,
        "payment_status": "paid|unpaid",
        "can_exit": true|false,
        "barrier_action": "OPEN|CLOSED",
        "display_message": "Payment info..."
    }
    """
    try:
        data = request.get_json()
        
        face_base64 = data.get('face_image')
        plate_base64 = data.get('plate_image')
        device_id = data.get('device_id')
        
        if not face_base64 or not plate_base64:
            return jsonify({'success': False, 'message': 'Missing images'}), 400
        
        if not device_id:
            return jsonify({'success': False, 'message': 'Missing device_id'}), 400
        
        # Decode images
        face_image = decode_base64_image(face_base64)
        plate_image = decode_base64_image(plate_base64)
        
        if face_image is None or plate_image is None:
            return jsonify({'success': False, 'message': 'Invalid image format'}), 400
        
        # Create parking service
        db_session = Session()
        try:
            parking_service = ParkingService(
                db_session,
                face_detector,
                face_recognizer,
                plate_detector
            )
            
            # Process check-out
            result = parking_service.process_check_out(
                face_image,
                plate_image,
                uuid.UUID(device_id)
            )
            
            db_session.commit()
            return jsonify(result)
            
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/parking/session/<license_plate>', methods=['GET'])
def get_session_by_plate(license_plate):
    """
    Get parking session info by license plate (for web payment)
    
    Response:
    {
        "session_id": "uuid",
        "license_plate": "51A-123.45",
        "check_in_time": "2024-01-20T10:00:00",
        "total_minutes": 120,
        "parking_fee": 20000.00
    }
    """
    try:
        db_session = Session()
        try:
            parking_service = ParkingService(db_session)
            result = parking_service.get_session_by_plate(license_plate)
            
            if result:
                return jsonify({'success': True, **result})
            else:
                return jsonify({'success': False, 'message': 'Session not found'}), 404
                
        finally:
            db_session.close()
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    # Run Flask server
    # For production, use gunicorn or uwsgi
    app.run(host='0.0.0.0', port=5000, debug=True)
