import paho.mqtt.client as mqtt
import json
import cv2
import os
import time
import ssl
import cloudinary
import cloudinary.uploader
import uuid
from datetime import datetime, timezone

#  CLOUDINARY 
cloudinary.config(
    cloud_name="motoguard",
    api_key="711384225714966",
    api_secret="MIVAF9tZKhYLvuLnsu2BypzxSbk"
)

def clean_ip(url):
    try:
        url = url.replace("http://", "").replace("https://", "")

        if "@" in url:
            url = url.split("@")[1]

        ip = url.split(":")[0]

        return ip
    except:
        return url

#  CAMERA CONFIG 
CAMERA_CONFIG = {
    "Bãi Xe Đại Học FPT": {
        "facein": "http://admin:admin@192.168.137.129:8081/video",
        "platein": "http://admin:admin@192.168.137.232:8081/video",
        "faceout": "http://admin:admin@192.168.137.204:8081/video",
        "plateout": "http://admin:admin@192.168.137.132:8081/video"
    }
}

#  MQTT CONFIG 
BROKER_IP = "l112e911.ala.asia-southeast1.emqxsl.com"
PORT = 8883

MQTT_USERNAME = "tien2908"
MQTT_PASSWORD = "tien2908"
TOPIC_CHECKIN = "parking/checkin"
TOPIC_CHECKOUT = "parking/checkout"
TOPIC_CONFIG = "parking/config/update"

client = mqtt.Client(client_id=f"parking_ai_{uuid.uuid4().hex[:6]}")

client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)

#  CONNECT 
def on_connect(client, userdata, flags, rc):
    global connected
    connected = True

    print("MQTT CONNECT RC =", rc)

    client.subscribe(TOPIC_CONFIG)
    client.subscribe("parking/responses")
    client.subscribe("parking/connection")

    send_camera_status("face_in")
    send_camera_status("plate_in")
    send_camera_status("face_out")
    send_camera_status("plate_out")

def on_message(client, userdata, msg):
    print(f"[AI MQTT] topic={msg.topic} payload={msg.payload.decode()}")
    if msg.topic == "parking/connection":
        try:
            data = json.loads(msg.payload.decode())
        except:
            data = None

        if data and data.get("eventName") == "TEST_CONNECTION" and data.get("status") == "CONNECTED":
            print(" TEST CONNECTION (CAMERA) ")

            send_camera_status("face_in")
            send_camera_status("plate_in")
            send_camera_status("face_out")
            send_camera_status("plate_out")

        return
    try:
        payload = msg.payload.decode()
        print("[MQTT RECEIVED]", msg.topic, payload)

        try:
            data = json.loads(payload)
        except:
            data = None

        if msg.topic == TOPIC_CONFIG:
            print("[CONFIG RECEIVED]", data)
            if config_callback and data:
                config_callback(data)
                
            return 

        if msg.topic == "parking/responses":
            if response_callback:
                response_callback(payload)
            return

    except Exception as e:
        print("MQTT ERROR:", e)

client.on_connect = on_connect
client.on_message = on_message

def register_config_callback(callback):
    client.config_callback = callback

def ensure_connected():
    if not client.is_connected():
        try:
            client.reconnect()
        except:
            client.connect(BROKER_IP, PORT, 60)

#  CLOUDINARY 
def upload_to_cloudinary(image, folder, prefix):
    if image is None:
        return None

    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        return None

    result = cloudinary.uploader.upload(
        buffer.tobytes(),
        folder=folder,
        public_id=f"{prefix}_{int(time.time())}"
    )

    return result.get("secure_url")

#  MQTT SEND 
def publish_with_retry(topic, payload, retries=3):
    for i in range(retries):
        try:
            ensure_connected()
            res = client.publish(topic, json.dumps(payload), qos=1)
            if res.rc == 0:
                print("MQTT SENT")
                return True
        except Exception as e:
            print("MQTT ERROR:", e)
        time.sleep(1)
    return False

#  CAMERA SERIAL 
SERIAL_FILE = "camera_serial.json"

def load_serial():
    if os.path.exists(SERIAL_FILE):
        with open(SERIAL_FILE, "r") as f:
            return json.load(f)

    serials = {
        k: str(uuid.uuid4())[:6]
        for k in ["face_in", "plate_in", "face_out", "plate_out"]
    }

    with open(SERIAL_FILE, "w") as f:
        json.dump(serials, f, indent=4)

    return serials

serials = load_serial()

#  CAMERA DEVICES 
CAMERA_DEVICES = {
    "face_in": {
        "device_code": f"face_camera_in_{serials['face_in']}",
        "device_name": "Face Camera In",
        "cameraIp": clean_ip(CAMERA_CONFIG["Bãi Xe Đại Học FPT"]["facein"])
    },
    "plate_in": {
        "device_code": f"plate_camera_in_{serials['plate_in']}",
        "device_name": "Plate Camera In",
        "cameraIp": clean_ip(CAMERA_CONFIG["Bãi Xe Đại Học FPT"]["platein"])
    },
    "face_out": {
        "device_code": f"face_camera_out_{serials['face_out']}",
        "device_name": "Face Camera Out",
        "cameraIp": clean_ip(CAMERA_CONFIG["Bãi Xe Đại Học FPT"]["faceout"])
    },
    "plate_out": {
        "device_code": f"plate_camera_out_{serials['plate_out']}",
        "device_name": "Plate Camera Out",
        "cameraIp": clean_ip(CAMERA_CONFIG["Bãi Xe Đại Học FPT"]["plateout"])
    }
}

#  CAMERA 
def send_camera_event(key):
    cam = CAMERA_DEVICES[key]

    payload = {
        "device_code": cam["device_code"],
        "device_name": cam["device_name"],
        "device_type": "camera",
        "event_type": "Detect",
        "event_source": "AI",
        "event_status": "Success",
        "ts": int(time.time())
    }

    topic = f"parking/{cam['device_code']}/event"
    publish_with_retry(topic, payload)

def send_camera_status(key):
    cam = CAMERA_DEVICES[key]

    payload = {
        "device_code": cam["device_code"],
        "device_name": cam["device_name"],
        "device_type": "camera",
        "cameraIp": cam["cameraIp"],
        "status": "online",
        "ts": int(time.time())
    }

    topic = f"parking/{cam['device_code']}/status"

    print("[MQTT SEND]", topic)
    print("[PAYLOAD]", payload)

    publish_with_retry(topic, payload)

#  CHECK-IN 
def send_checkin(plate_number, face_img, plate_img,
                 status, reason, lot_name,
                 confidence_score, processing_time_ms):

    send_camera_event("face_in")
    send_camera_event("plate_in")

    face_url = upload_to_cloudinary(face_img, "checkin/faces", plate_number)
    plate_url = upload_to_cloudinary(plate_img, "checkin/plates", plate_number)

    cam = CAMERA_CONFIG[lot_name]

    payload = {
        "event": "checkin",
        "plateNumber": plate_number,
        "status": status,
        "reason": reason,
        "timeIn": datetime.now(timezone.utc).isoformat(),
        "faceCameraIp": clean_ip(cam["facein"]),
        "plateCameraIp": clean_ip(cam["platein"]),
        "faceImageUrl": face_url,
        "plateImageUrl": plate_url,
        "confidenceScore": confidence_score,
        "processingTimeMs": processing_time_ms
    }

    publish_with_retry(TOPIC_CHECKIN, payload)

#  CHECK-OUT 
def send_checkout(plate_number, similarity,
                  face_img, plate_img,
                  status, reason, lot_name,
                  confidence_score, processing_time_ms):

    send_camera_event("face_out")
    send_camera_event("plate_out")

    face_url = upload_to_cloudinary(face_img, "checkout/faces", plate_number)
    plate_url = upload_to_cloudinary(plate_img, "checkout/plates", plate_number)

    cam = CAMERA_CONFIG[lot_name]

    payload = {
        "event": "checkout",
        "plateNumber": plate_number,
        "status": status,
        "reason": reason,
        "similarity": similarity,
        "timeOut": datetime.now(timezone.utc).isoformat(),
        "faceCameraIp": clean_ip(cam["faceout"]),
        "plateCameraIp": clean_ip(cam["plateout"]),
        "faceImageUrl": face_url,
        "plateImageUrl": plate_url,
        "confidenceScore": confidence_score,
        "processingTimeMs": processing_time_ms
    }

    publish_with_retry(TOPIC_CHECKOUT, payload)
response_callback = None

def register_response_callback(callback):
    global response_callback
    response_callback = callback

config_callback = None

def register_config_callback(callback):
    global config_callback
    config_callback = callback
#  INIT 
client.connect(BROKER_IP, PORT, 60)
client.loop_start()
