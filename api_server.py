"""Minimal AI embedding service with backend callback worker."""

from io import BytesIO
import base64
import os
import threading
import time

import cv2
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from PIL import Image
import requests
import urllib3

from face_recognition.face_detection import FaceDetector
from face_recognition.face_recognition import FaceRecognizer

load_dotenv()

app = Flask(__name__)
CORS(app)

print("Initializing AI models...")
face_detector = FaceDetector()
face_recognizer = FaceRecognizer()
print("AI embedding service ready")

BE_BASE_URL = os.getenv("BE_BASE_URL", "https://sep490motoguard-production.up.railway.app").rstrip("/")
AI_POLL_INTERVAL_SECONDS = float(os.getenv("AI_POLL_INTERVAL_SECONDS", "0.5"))
VERIFY_BE_SSL = os.getenv("VERIFY_BE_SSL", "false").lower() in ("1", "true", "yes")

if not VERIFY_BE_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def decode_base64_image(base64_string):
    """Decode base64 image to OpenCV BGR array."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        img_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(img_data))
        img_array = np.array(img)

        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        return img_array
    except Exception as ex:
        print(f"Error decoding image: {ex}")
        return None


def extract_embedding_from_base64(image_base64):
    """Extract embedding bytes encoded as base64 from image payload."""
    image = decode_base64_image(image_base64)
    if image is None:
        raise ValueError("Invalid image format")

    faces = face_detector.detect_faces(image)
    if not faces:
        raise ValueError("No face detected")

    x, y, w, h = faces[0]["box"]
    y_min, y_max = max(0, y), min(image.shape[0], y + h)
    x_min, x_max = max(0, x), min(image.shape[1], x + w)
    face_crop = image[y_min:y_max, x_min:x_max]
    face_crop = cv2.resize(face_crop, (160, 160))

    embedding = face_recognizer.get_embedding(face_crop)
    embedding_bytes = embedding.astype(np.float32).tobytes()
    return base64.b64encode(embedding_bytes).decode("utf-8")


def poll_embedding_jobs():
    """Background worker: pull embedding jobs from backend and push result back."""
    print(f"Starting embedding callback worker against {BE_BASE_URL} (verify_ssl={VERIFY_BE_SSL})")

    while True:
        try:
            next_url = f"{BE_BASE_URL}/api/v1/ai/embedding-jobs/next"
            response = requests.get(next_url, timeout=10, verify=VERIFY_BE_SSL)

            if response.status_code == 204:
                time.sleep(AI_POLL_INTERVAL_SECONDS)
                continue

            if response.status_code != 200:
                print(f"Poll failed: HTTP {response.status_code} - {response.text}")
                time.sleep(2)
                continue

            payload = response.json()
            if not payload.get("success"):
                time.sleep(AI_POLL_INTERVAL_SECONDS)
                continue

            data = payload.get("data") or {}
            job_id = data.get("jobId")
            image_base64 = data.get("image")

            if not job_id or not image_base64:
                time.sleep(AI_POLL_INTERVAL_SECONDS)
                continue

            result_body = {}
            try:
                result_body["embedding"] = extract_embedding_from_base64(image_base64)
            except Exception as ex:
                result_body["errorMessage"] = str(ex)

            submit_url = f"{BE_BASE_URL}/api/v1/ai/embedding-jobs/{job_id}/result"
            submit_response = requests.post(submit_url, json=result_body, timeout=10, verify=VERIFY_BE_SSL)

            if submit_response.status_code != 200:
                print(
                    "Submit result failed for job "
                    f"{job_id}: HTTP {submit_response.status_code} - {submit_response.text}"
                )

        except Exception as ex:
            print(f"Embedding worker error: {ex}")
            time.sleep(2)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "services": {
                "face_detection": face_detector is not None,
                "face_recognition": face_recognizer is not None,
            },
        }
    )


@app.route("/api/extract/embedding", methods=["POST"])
def extract_face_embedding():
    try:
        data = request.get_json() or {}
        image_base64 = data.get("image")

        if not image_base64:
            return jsonify({"success": False, "message": "No image provided"}), 400

        embedding_base64 = extract_embedding_from_base64(image_base64)

        return jsonify(
            {
                "success": True,
                "embedding": embedding_base64,
                "dim": 512,
            }
        )
    except Exception as ex:
        print(f"Embedding extraction error: {ex}")
        return jsonify({"success": False, "message": str(ex)}), 500


if __name__ == "__main__":
    worker_thread = threading.Thread(target=poll_embedding_jobs, daemon=True)
    worker_thread.start()
    app.run(host="0.0.0.0", port=5000, debug=True)
