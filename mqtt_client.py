import paho.mqtt.client as mqtt
import json
import os
import cv2
from datetime import datetime

BROKER_IP = "192.168.2.131"
PORT = 1883

TOPIC_CHECKIN = "parking/checkin"
TOPIC_CHECKOUT = "parking/checkout"

CHECKIN_DIR = r"D:\Code\Model_Camera\parking_images\checkin"
CHECKOUT_DIR = r"D:\Code\Model_Camera\parking_images\checkout"

#Link lưu BE nha Tiến
NETWORK_PATH = r"\\192.168.2.131\parking_images"

os.makedirs(CHECKIN_DIR, exist_ok=True)
os.makedirs(CHECKOUT_DIR, exist_ok=True)

client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ MQTT Connected:", BROKER_IP)
    else:
        print("❌ MQTT Connection failed:", rc)

client.on_connect = on_connect
client.connect(BROKER_IP, PORT, 60)
client.loop_start()

def save_image(image, path):

    if image is None:
        return None

    cv2.imwrite(path, image)
    return path

def send_checkin(plate_number, face_img, plate_img, camera_ip):

    local_face_path = f"{CHECKIN_DIR}\\{plate_number}_face.jpg"
    local_plate_path = f"{CHECKIN_DIR}\\{plate_number}_plate.jpg"

    network_face_path = f"{NETWORK_PATH}\\checkin\\{plate_number}_face.jpg"
    network_plate_path = f"{NETWORK_PATH}\\checkin\\{plate_number}_plate.jpg"

    if face_img is not None:
        save_image(face_img, local_face_path)

    if plate_img is not None:
        save_image(plate_img, local_plate_path)
    #SenT MQTT check-in event
    data = {
        "plate_number": plate_number,
        "camera_ip": camera_ip,
        "time_in": datetime.now().isoformat(),
        "face_image": network_face_path,
        "plate_image": network_plate_path
    }

    client.publish(TOPIC_CHECKIN, json.dumps(data))

    print("📡 MQTT CHECKIN SENT")
    print(data)

def send_checkout(
    plate_number,
    similarity,
    camera_ip,
    face_img=None,
    plate_img=None
):

    local_face_path = f"{CHECKOUT_DIR}\\{plate_number}_face.jpg"
    local_plate_path = f"{CHECKOUT_DIR}\\{plate_number}_plate.jpg"

    network_face_path = f"{NETWORK_PATH}\\checkout\\{plate_number}_face.jpg"
    network_plate_path = f"{NETWORK_PATH}\\checkout\\{plate_number}_plate.jpg"

    if face_img is not None:
        save_image(face_img, local_face_path)

    if plate_img is not None:
        save_image(plate_img, local_plate_path)
    #SenT MQTT check-out event
    data = {
        "plate_number": plate_number,
        "camera_ip": camera_ip,
        "time_out": datetime.now().isoformat(),
        "similarity": similarity,
        "face_image": network_face_path,
        "plate_image": network_plate_path
    }

    client.publish(TOPIC_CHECKOUT, json.dumps(data))

    print("📡 MQTT CHECKOUT SENT")
    print(data)