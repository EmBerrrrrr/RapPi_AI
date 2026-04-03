from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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

        print("\n===== SCRIPT LOG =====")
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


# ================= CHECKIN =================
@app.route('/checkin')
def checkin():
    print("[VEHICLE CHECKIN]")

    script_path = os.path.join(BASE_DIR, "camera", "checkin_capture.py")

    result = subprocess.run(
        ["python", script_path],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        return jsonify({"status": "OPEN"})
    else:
        return jsonify({"status": "DENY"})

# ================= CHECKOUT =================
@app.route('/checkout')
def checkout():
    script_path = os.path.join(BASE_DIR, "camera", "checkout_capture.py")
    result = run_script(["python", script_path])

    if result == "OPEN":
        return jsonify({"status": "OPEN"})
    else:
        return jsonify({"status": "DENY"})
    
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

    print("=== ROUTES ===")
    for rule in app.url_map.iter_rules():
        print(rule)

    app.run(host="0.0.0.0", port=5000, threaded=True)