"""
MQTT Client for MotoGuard Parking System
Uploads images to Cloudinary and sends URLs via MQTT to backend
"""

import paho.mqtt.client as mqtt
import json
import os
import cv2
import ssl
import cloudinary
import cloudinary.uploader
from datetime import datetime, timezone

# ============================================================================
# Cloudinary Configuration
# ============================================================================
# Get these from: https://cloudinary.com/console
cloudinary.config(
    cloud_name="motoguard",      # Replace with your cloud name
    api_key="711384225714966",   # Replace with your API key
    api_secret="MIVAF9tZKhYLvuLnsu2BypzxSbk"  # Replace with your API secret
)

# ============================================================================
# EMQX Cloud Broker Configuration
# ============================================================================
BROKER_IP = "l112e911.ala.asia-southeast1.emqxsl.com"
PORT = 8883  # MQTT over TLS/SSL
USE_TLS = True

# EMQX Authentication
MQTT_USERNAME = "tien2908"
MQTT_PASSWORD = "tien2908"

# MQTT Topics
TOPIC_CHECKIN = "parking/checkin"
TOPIC_CHECKOUT = "parking/checkout"

# ============================================================================
# Parking Lot Configuration (Get from database)
# ============================================================================
# TODO: Replace with actual parking lot GUID from your database
DEFAULT_LOT_ID = "0c3b5fb8-a45b-4726-b2b3-a0c3a0ae25b8"

# Optional: Gate ID if using specific gates
DEFAULT_GATE_ID = None  # or "gate-guid-here"

# ============================================================================
# Local Storage Paths (for debugging)
# ============================================================================
CHECKIN_DIR = r"D:\Code\Model_Camera\parking_images\checkin"
CHECKOUT_DIR = r"D:\Code\Model_Camera\parking_images\checkout"

os.makedirs(CHECKIN_DIR, exist_ok=True)
os.makedirs(CHECKOUT_DIR, exist_ok=True)

# ============================================================================
# MQTT Client Setup
# ============================================================================
client = mqtt.Client(client_id="parking_system", clean_session=True, protocol=mqtt.MQTTv311)

# Set username and password for EMQX authentication
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

# Configure TLS/SSL for EMQX Cloud
if USE_TLS:
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)
    # Uncomment below to skip hostname verification (not recommended for production)
    # client.tls_insecure_set(True)

def on_connect(client, userdata, flags, rc):
    """Callback when connected to MQTT broker"""
    if rc == 0:
        print("✅ EMQX Cloud Connected:", BROKER_IP)
        print(f"📡 Topics: {TOPIC_CHECKIN}, {TOPIC_CHECKOUT}")
    else:
        print(f"❌ EMQX Connection failed with code: {rc}")
        if rc == 5:
            print("   → Check username/password")
        elif rc == 1:
            print("   → Incorrect protocol version")
        elif rc == 3:
            print("   → Server unavailable")

client.on_connect = on_connect

# Connect to EMQX broker
client.connect(BROKER_IP, PORT, 60)
client.loop_start()

# ============================================================================
# Helper Functions
# ============================================================================

def save_image(image, path):
    """Save OpenCV image to local path for debugging"""
    if image is None:
        return None
    cv2.imwrite(path, image)
    return path

def upload_to_cloudinary(image, folder="parking-sessions", filename_prefix="mqtt"):
    """
    Upload OpenCV image to Cloudinary
    
    Args:
        image: OpenCV image (numpy array)
        folder: Cloudinary folder path
        filename_prefix: Prefix for filename
        
    Returns:
        str: Cloudinary URL or None if failed
    """
    if image is None:
        return None
    
    try:
        # Encode image to JPEG in memory
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            print("⚠️  Failed to encode image to JPEG")
            return None
        
        # Upload to Cloudinary
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        public_id = f"{filename_prefix}_{timestamp}"
        
        result = cloudinary.uploader.upload(
            buffer.tobytes(),
            folder=folder,
            public_id=public_id,
            resource_type="image",
            format="jpg"
        )
        
        # Return secure URL
        return result.get('secure_url')
        
    except Exception as e:
        print(f"❌ Cloudinary upload failed: {e}")
        return None

# ============================================================================
# Main Functions
# ============================================================================

def send_checkin(
    plate_number,
    face_img=None,
    plate_img=None,
    camera_ip=None,
    lot_id=None,
    gate_id=None
):
    """
    Send check-in event to backend via MQTT
    Images are uploaded to Cloudinary first, then URLs are sent
    
    Args:
        plate_number (str): License plate number (required)
        face_img (numpy.ndarray): Face image from camera (optional)
        plate_img (numpy.ndarray): License plate image from camera (optional)
        camera_ip (str): IP address of camera (optional)
        lot_id (str): Parking lot GUID (optional, uses DEFAULT_LOT_ID if not provided)
        gate_id (str): Entry gate GUID (optional)
    """
    # Use default lot_id if not provided
    if lot_id is None:
        lot_id = DEFAULT_LOT_ID
    
    # Save images locally for debugging
    if face_img is not None:
        local_face_path = os.path.join(CHECKIN_DIR, f"{plate_number}_face.jpg")
        save_image(face_img, local_face_path)
    
    if plate_img is not None:
        local_plate_path = os.path.join(CHECKIN_DIR, f"{plate_number}_plate.jpg")
        save_image(plate_img, local_plate_path)
    
    # Upload images to Cloudinary and get URLs
    print("📤 Uploading images to Cloudinary...")
    face_url = upload_to_cloudinary(face_img, folder="parking-sessions/mqtt/faces", filename_prefix=f"checkin_{plate_number}")
    plate_url = upload_to_cloudinary(plate_img, folder="parking-sessions/mqtt/plates", filename_prefix=f"checkin_{plate_number}")
    
    # Prepare MQTT payload (camelCase for C# compatibility)
    payload = {
        "lotId": lot_id,
        "plateNumber": plate_number,
        "timeIn": datetime.now(timezone.utc).isoformat(),  # UTC time with timezone
        "cameraIp": camera_ip,
        "faceImageUrl": face_url,      # Cloudinary URL (preferred)
        "plateImageUrl": plate_url,    # Cloudinary URL (preferred)
        "gateId": gate_id
    }
    
    # Publish to MQTT
    result = client.publish(TOPIC_CHECKIN, json.dumps(payload), qos=1)
    
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print("✅ CHECKIN SENT")
        print(f"   Plate: {plate_number}")
        print(f"   Lot ID: {lot_id}")
        print(f"   Face URL: {face_url if face_url else 'None'}")
        print(f"   Plate URL: {plate_url if plate_url else 'None'}")
    else:
        print(f"❌ CHECKIN FAILED: {result.rc}")

def send_checkout(
    plate_number,
    similarity=None,
    camera_ip=None,
    face_img=None,
    plate_img=None,
    lot_id=None,
    gate_id=None
):
    """
    Send check-out event to backend via MQTT
    Images are uploaded to Cloudinary first, then URLs are sent
    
    Args:
        plate_number (str): License plate number (required)
        similarity (float): Face similarity score from recognition (optional)
        camera_ip (str): IP address of camera (optional)
        face_img (numpy.ndarray): Face image from camera (optional)
        plate_img (numpy.ndarray): License plate image from camera (optional)
        lot_id (str): Parking lot GUID (optional, uses DEFAULT_LOT_ID if not provided)
        gate_id (str): Exit gate GUID (optional)
    """
    # Use default lot_id if not provided
    if lot_id is None:
        lot_id = DEFAULT_LOT_ID
    
    # Save images locally for debugging
    if face_img is not None:
        local_face_path = os.path.join(CHECKOUT_DIR, f"{plate_number}_face.jpg")
        save_image(face_img, local_face_path)
    
    if plate_img is not None:
        local_plate_path = os.path.join(CHECKOUT_DIR, f"{plate_number}_plate.jpg")
        save_image(plate_img, local_plate_path)
    
    # Upload images to Cloudinary and get URLs
    print("📤 Uploading images to Cloudinary...")
    face_url = upload_to_cloudinary(face_img, folder="parking-sessions/mqtt/faces_checkout", filename_prefix=f"checkout_{plate_number}")
    plate_url = upload_to_cloudinary(plate_img, folder="parking-sessions/mqtt/plates_checkout", filename_prefix=f"checkout_{plate_number}")
    
    # Prepare MQTT payload (camelCase for C# compatibility)
    payload = {
        "lotId": lot_id,
        "plateNumber": plate_number,
        "timeOut": datetime.now(timezone.utc).isoformat(),  # UTC time with timezone
        "cameraIp": camera_ip,
        "similarity": similarity,
        "faceImageUrl": face_url,      
        "plateImageUrl": plate_url,    # Cloudinary URL (preferred)
        "gateId": gate_id
    }
    
    # Publish to MQTT
    result = client.publish(TOPIC_CHECKOUT, json.dumps(payload), qos=1)
    
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print("✅ CHECKOUT SENT")
        print(f"   Plate: {plate_number}")
        print(f"   Lot ID: {lot_id}")
        print(f"   Similarity: {similarity}")
        print(f"   Face URL: {face_url if face_url else 'None'}")
        print(f"   Plate URL: {plate_url if plate_url else 'None'}")
    else:
        print(f"❌ CHECKOUT FAILED: {result.rc}")

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import time
    
    print("\n" + "="*60)
    print("MotoGuard MQTT Client - Test Mode")
    print("Uploads images to Cloudinary, sends URLs via MQTT")
    print("="*60 + "\n")
    
    # Wait for connection
    time.sleep(2)
    
    # Example: Load test images (replace with actual camera capture)
    # face_image = cv2.imread("test_face.jpg")
    # plate_image = cv2.imread("test_plate.jpg")
    
    # Example 1: Check-in without images (plate number only)
    print("\n📍 Test 1: Check-in (plate only)")
    send_checkin(
        plate_number="ABC123",
        camera_ip="192.168.1.100",
        lot_id="your-lot-guid-here"  # Replace with actual GUID from database
    )
    
    time.sleep(1)
    
    # Example 2: Check-in with images
    print("\n📍 Test 2: Check-in (with images)")
    # Uncomment when you have images:
    # send_checkin(
    #     plate_number="XYZ789",
    #     face_img=face_image,
    #     plate_img=plate_image,
    #     camera_ip="192.168.1.100",
    #     lot_id="your-lot-guid-here"
    # )
    
    time.sleep(1)
    
    # Example 3: Check-out
    print("\n📍 Test 3: Check-out")
    send_checkout(
        plate_number="ABC123",
        similarity=0.95,
        camera_ip="192.168.1.100",
        lot_id="your-lot-guid-here"  # Replace with actual GUID from database
    )
    
    print("\n" + "="*60)
    print("Tests completed. Press Ctrl+C to exit.")
    print("NOTE: Install cloudinary: pip install cloudinary")
    print("="*60 + "\n")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Disconnecting...")
        client.loop_stop()
        client.disconnect()
