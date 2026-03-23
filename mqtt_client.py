"""
MQTT Client for MotoGuard Parking System (FIXED VERSION)
Supports success/fail status for barrier control
"""

import paho.mqtt.client as mqtt
import json
import os
import cv2
import ssl
import cloudinary
import cloudinary.uploader
from datetime import datetime, timezone

# ================= CLOUDINARY =================
cloudinary.config(
    cloud_name="motoguard",
    api_key="711384225714966",
    api_secret="MIVAF9tZKhYLvuLnsu2BypzxSbk"
)

# ================= MQTT CONFIG =================
BROKER_IP = "l112e911.ala.asia-southeast1.emqxsl.com"
PORT = 8883
USE_TLS = True

MQTT_USERNAME = "tien2908"
MQTT_PASSWORD = "tien2908"

TOPIC_CHECKIN = "parking/checkin"
TOPIC_CHECKOUT = "parking/checkout"

DEFAULT_LOT_ID = "0c3b5fb8-a45b-4726-b2b3-a0c3a0ae25b8"
DEFAULT_GATE_ID = None

CHECKIN_DIR = r"D:\Code\Model_Camera\parking_images\checkin"
CHECKOUT_DIR = r"D:\Code\Model_Camera\parking_images\checkout"

os.makedirs(CHECKIN_DIR, exist_ok=True)
os.makedirs(CHECKOUT_DIR, exist_ok=True)

# ================= MQTT CLIENT =================
client = mqtt.Client(client_id="parking_system", clean_session=True)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

if USE_TLS:
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ EMQX Connected")
    else:
        print(f"❌ MQTT Connection failed: {rc}")

client.on_connect = on_connect
client.connect(BROKER_IP, PORT, 60)
client.loop_start()

# ================= HELPER =================
def save_image(image, path):
    if image is None:
        return None
    cv2.imwrite(path, image)
    return path

def upload_to_cloudinary(image, folder, prefix):
    if image is None:
        return None

    try:
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            return None

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

        result = cloudinary.uploader.upload(
            buffer.tobytes(),
            folder=folder,
            public_id=f"{prefix}_{timestamp}"
        )

        return result.get("secure_url")

    except Exception as e:
        print(f"❌ Cloudinary error: {e}")
        return None

# ================= CHECK-IN =================
def send_checkin(
    plate_number,
    face_img=None,
    plate_img=None,
    camera_ip=None,
    lot_id=None,
    gate_id=None,
    status="success",     # ✅ NEW
    reason="ok"           # ✅ NEW
):
    if lot_id is None:
        lot_id = DEFAULT_LOT_ID

    print(f"\n📤 CHECK-IN MQTT ({status.upper()})")

    # Upload ảnh
    face_url = upload_to_cloudinary(face_img, "parking/checkin/faces", f"{plate_number}_face")
    plate_url = upload_to_cloudinary(plate_img, "parking/checkin/plates", f"{plate_number}_plate")

    payload = {
        "event": "checkin",
        "status": status,
        "reason": reason,

        "lotId": lot_id,
        "plateNumber": plate_number,
        "timeIn": datetime.now(timezone.utc).isoformat(),
        "cameraIp": camera_ip,
        "faceImageUrl": face_url,
        "plateImageUrl": plate_url,
        "gateId": gate_id
    }

    result = client.publish(TOPIC_CHECKIN, json.dumps(payload), qos=1)

    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print(f"✅ CHECK-IN SENT ({status}) - {plate_number}")
    else:
        print(f"❌ CHECK-IN FAILED: {result.rc}")

# ================= CHECK-OUT =================
def send_checkout(
    plate_number,
    similarity=None,
    camera_ip=None,
    face_img=None,
    plate_img=None,
    lot_id=None,
    gate_id=None,
    status="success",   # ✅ NEW
    reason="ok"         # ✅ NEW
):
    if lot_id is None:
        lot_id = DEFAULT_LOT_ID

    print(f"\n📤 CHECK-OUT MQTT ({status.upper()})")

    face_url = upload_to_cloudinary(face_img, "parking/checkout/faces", f"{plate_number}_face")
    plate_url = upload_to_cloudinary(plate_img, "parking/checkout/plates", f"{plate_number}_plate")

    payload = {
        "event": "checkout",
        "status": status,
        "reason": reason,

        "lotId": lot_id,
        "plateNumber": plate_number,
        "timeOut": datetime.now(timezone.utc).isoformat(),
        "cameraIp": camera_ip,
        "similarity": similarity,
        "faceImageUrl": face_url,
        "plateImageUrl": plate_url,
        "gateId": gate_id
    }

    result = client.publish(TOPIC_CHECKOUT, json.dumps(payload), qos=1)

    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print(f"✅ CHECK-OUT SENT ({status}) - {plate_number}")
    else:
        print(f"❌ CHECK-OUT FAILED: {result.rc}")