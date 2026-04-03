import paho.mqtt.client as mqtt
import json
import cv2
import time
import ssl
import cloudinary
import cloudinary.uploader
import uuid
from datetime import datetime, timezone

# CLOUDINARY
cloudinary.config(
    cloud_name="motoguard",
    api_key="711384225714966",
    api_secret="MIVAF9tZKhYLvuLnsu2BypzxSbk"
)

# MQTT CONFIG
BROKER_IP = "l112e911.ala.asia-southeast1.emqxsl.com"
PORT = 8883

MQTT_USERNAME = "tien2908"
MQTT_PASSWORD = "tien2908"

TOPIC_CHECKIN = "parking/checkin"
TOPIC_CHECKOUT = "parking/checkout"

client = mqtt.Client(
    client_id=f"parking_system_{uuid.uuid4().hex[:6]}",
    clean_session=True
)

client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)


def on_connect(client, userdata, flags, rc):
    print("MQTT CONNECT RC =", rc)


client.on_connect = on_connect


def ensure_connected():
    if not client.is_connected():
        try:
            client.reconnect()
        except:
            client.connect(BROKER_IP, PORT, 60)


client.connect(BROKER_IP, PORT, 60)
client.loop_start()


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
        print("Cloudinary error:", e)
        return None


def publish_with_retry(topic, payload, retries=3, delay=1):
    for attempt in range(retries):
        try:
            ensure_connected()
            result = client.publish(topic, json.dumps(payload), qos=1)

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"MQTT SENT (attempt {attempt+1})")
                return True

        except Exception as e:
            print(f"MQTT ERROR (attempt {attempt+1}):", e)

        time.sleep(delay)

    print("MQTT FAILED AFTER RETRY")
    return False


# CHECK-IN
def send_checkin(plate_number, face_img=None, plate_img=None, camera_ip=None,
                 status="success", reason="ok"):

    face_url = upload_to_cloudinary(face_img, "parking/checkin/faces", f"{plate_number}_face")
    plate_url = upload_to_cloudinary(plate_img, "parking/checkin/plates", f"{plate_number}_plate")

    payload = {
        "event": "checkin",
        "status": status,
        "reason": reason,
        "plateNumber": plate_number,
        "timeIn": datetime.now(timezone.utc).isoformat(),
        "cameraIp": camera_ip,
        "faceImageUrl": face_url,
        "plateImageUrl": plate_url
    }

    success = publish_with_retry(TOPIC_CHECKIN, payload)

    if success:
        print("CHECK-IN SENT")
    else:
        print("CHECK-IN LOST")


# CHECK-OUT
def send_checkout(plate_number, similarity, camera_ip,
                  face_img, plate_img, status, reason):

    face_url = upload_to_cloudinary(face_img, "parking/checkout/faces", f"{plate_number}_face")
    plate_url = upload_to_cloudinary(plate_img, "parking/checkout/plates", f"{plate_number}_plate")
    payload = {
        "event": "checkout",
        "plateNumber": plate_number,
        "status": status,
        "reason": reason,
        "similarity": similarity,
        "cameraIp": camera_ip,
        "faceImageUrl": face_url,
        "plateImageUrl": plate_url
    }

    publish_with_retry("parking/checkout", payload)