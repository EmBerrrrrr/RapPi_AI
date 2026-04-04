from flask import Flask, request, jsonify
import subprocess
import os
import threading
import mqtt_client

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

        print("\n SCRIPT LOG ")
        print(result.stdout)
        print(result.stderr)
        print("[RETURN CODE]:", result.returncode)

        if result.returncode == 0:
            return "OPEN"
        elif result.returncode == 2:
            return "DENY"
        else:
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


#  THREAD CHECKOUT 
def run_checkout_ai():
    global checkout_status

    script_path = os.path.join(BASE_DIR, "camera", "checkout_capture.py")

    run_script(["python", script_path])

    checkout_status = {
        "status": "DONE"
    }


#  CHECKIN 
@app.route('/checkin')
def checkin():
    global checkin_status

    print("[VEHICLE CHECKIN]")

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

    checkout_status = {"status": "PROCESSING"}  

    threading.Thread(target=run_checkout_ai).start()

    return jsonify({"status": "TRIGGERED"})


# FACE CHECKIN 
@app.route('/face_checkin', methods=['POST'])
def face_checkin():
    token = (request.json or {}).get("token", "")

    script_path = os.path.join(BASE_DIR, "camera", "face_checkin_single_cam.py")

    result = run_script([
        "python",
        script_path,
        "checkin",
        token
    ])

    return jsonify({"status": result})


# FACE CHECKOUT 
@app.route('/face_checkout', methods=['POST'])
def face_checkout():
    token = (request.json or {}).get("token", "")

    script_path = os.path.join(BASE_DIR, "camera", "face_checkin_single_cam.py")

    result = run_script([
        "python",
        script_path,
        "checkout",
        token
    ])

    return jsonify({"status": result})


if __name__ == "__main__":
    print("AI SERVER RUNNING PORT 5000")

    for rule in app.url_map.iter_rules():
        print(rule)

    mqtt_client.client.connect(mqtt_client.BROKER_IP, mqtt_client.PORT, 60)
    mqtt_client.client.loop_start()
    
    mqtt_client.send_camera_status("face_in")
    mqtt_client.send_camera_status("plate_in")
    mqtt_client.send_camera_status("face_out")
    mqtt_client.send_camera_status("plate_out")

    app.run(host="0.0.0.0", port=5000, threaded=True)