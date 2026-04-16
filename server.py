from flask import Flask, request, jsonify
import subprocess
import os
import threading
import mqtt_client
import time
import json

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#  STORE RESULT 
checkin_status = {"status": "IDLE"}
checkout_status = {"status": "IDLE"}


#  RUN SCRIPT 
def run_script(command):
    try:
        print("\n[RUN]", command)

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            encoding='utf-8',
            errors='replace'
        )

        print("\n=== SCRIPT LOG ===")
        print(result.stdout)

        try:
            last_line = result.stdout.strip().split("\n")[-1]
            data = json.loads(last_line)

            # 🔥 FIX QUAN TRỌNG: phải return ở đây
            if data.get("success"):
                return "OPEN"
            else:
                return "DENY"

        except Exception as e:
            print("JSON parse failed:", e)
            print("RAW OUTPUT:", result.stdout)

            # 🔥 fallback thông minh
            if "MATCH SUCCESS" in result.stdout:
                return "OPEN"
            elif "MATCH FAIL" in result.stdout:
                return "DENY"

            return "ERROR"

    except Exception as e:
        print("ERROR:", e)
        return "ERROR"

#  THREAD CHECKIN 
def run_checkin_ai():
    global checkin_status

    script_path = os.path.join(BASE_DIR, "camera", "checkin_capture.py")
    result = run_script(["python", script_path])

    checkin_status = {"status": result}

#  HEARTBEAT CAMERA (gửi mỗi 5 phút)
def camera_heartbeat():
    time.sleep(60)

    while True:
        if not mqtt_client.connected:
            time.sleep(1)
            continue

        print("\n[HEARTBEAT] Sending camera status...")

        mqtt_client.send_camera_status("face_in")
        mqtt_client.send_camera_status("plate_in")
        mqtt_client.send_camera_status("face_out")
        mqtt_client.send_camera_status("plate_out")

        time.sleep(60)  # 300s = 5 phút

#  THREAD CHECKOUT 
def run_checkout_ai():
    global checkout_status

    script_path = os.path.join(BASE_DIR, "camera", "checkout_capture.py")

    result = run_script(["python", script_path])

    checkout_status = {
        "status": result 
    }

    print(f"[CHECKOUT RESULT]: {result}")


#  CHECKIN 
@app.route('/checkin')
def checkin():
    global checkin_status

    print("[VEHICLE CHECKIN]")

    # ===== CHỐNG SPAM THREAD =====
    if checkin_status.get("status") == "PROCESSING":
        print("CHECKIN BUSY → IGNORE")
        return jsonify({"status": "BUSY"})

    checkin_status = {"status": "PROCESSING"}

    threading.Thread(target=run_checkin_ai).start()

    return jsonify({"status": "PROCESSING"})


@app.route('/checkin_result')
def checkin_result():
    return jsonify(checkin_status)


#  CHECKOUT 
@app.route('/checkout')
def checkout():
    global checkout_status

    # ===== CHỐNG SPAM =====
    if checkout_status.get("status") == "PROCESSING":
        print("CHECKOUT BUSY → IGNORE")
        return jsonify({"status": "BUSY"})

    checkout_status = {"status": "PROCESSING"}  

    threading.Thread(target=run_checkout_ai).start()

    return jsonify({"status": "TRIGGERED"})

@app.route('/checkout_result')
def checkout_result():
    return jsonify(checkout_status)

# FACE CHECKIN 
@app.route('/face_checkin', methods=['POST'])
def face_checkin():
    token = (request.json or {}).get("token", "")

    script_path = os.path.join(BASE_DIR, "camera", "face_checkin_single_cam.py")

    print("[FACE CHECKIN] Trigger camera...")

    threading.Thread(
        target=run_script,
        args=([ "python", script_path, "checkin", token ],)
    ).start()

    return jsonify({"status": "PROCESSING"})


# FACE CHECKOUT 
@app.route('/face_checkout', methods=['POST'])
def face_checkout():
    token = (request.json or {}).get("token", "")

    script_path = os.path.join(BASE_DIR, "camera", "face_checkin_single_cam.py")

    print("[FACE CHECKOUT] Trigger camera...")

    threading.Thread(
        target=run_script,
        args=([ "python", script_path, "checkout", token ],)
    ).start()

    return jsonify({"status": "PROCESSING"})


if __name__ == "__main__":
    print("AI SERVER RUNNING PORT 5000")

    for rule in app.url_map.iter_rules():
        print(rule)

    mqtt_client.client.connect(mqtt_client.BROKER_IP, mqtt_client.PORT, 60)
    mqtt_client.client.loop_start()
    time.sleep(2)

    threading.Thread(target=camera_heartbeat, daemon=True).start()

    app.run(host="0.0.0.0", port=5000, threaded=True)