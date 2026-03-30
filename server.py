from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(command):
    try:
        print("\n[RUN]", command)

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )

        try:
            stdout, stderr = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            print("TIMEOUT")
            return False

        print("\n===== SCRIPT LOG =====")
        print(stdout)
        print(stderr)
        print("[RETURN CODE]:", process.returncode)

        return process.returncode == 0

    except Exception as e:
        print("ERROR:", e)
        return False

# ================= VEHICLE (GIỮ NGUYÊN) =================
@app.route('/checkin')
def checkin():
    print("[VEHICLE CHECKIN]")

    script_path = os.path.join(BASE_DIR, "camera", "checkin_capture.py")

    success = run_script([
        "python",
        script_path
    ])

    return "OPEN" if success else "DENY"


@app.route('/checkout')
def checkout():
    print("[VEHICLE CHECKOUT]")

    script_path = os.path.join(BASE_DIR, "camera", "checkout_capture.py")

    success = run_script([
        "python",
        script_path
    ])

    return "OPEN" if success else "DENY"


# ================= STAFF (FACE) =================
@app.route('/face_checkin', methods=['POST'])
def face_checkin():
    token = (request.json or {}).get("token", "")

    print("[FACE CHECKIN]")
    print("[TOKEN]:", token)

    script_path = os.path.join(BASE_DIR, "camera", "face_checkin_single_cam.py")

    success = run_script([
        "python",
        script_path,
        "checkin",
        token
    ])

    return jsonify({
        "status": "OPEN" if success else "DENY"
    })


@app.route('/face_checkout', methods=['POST'])
def face_checkout():
    token = (request.json or {}).get("token", "")

    print("[FACE CHECKOUT]")
    print("[TOKEN]:", token)

    script_path = os.path.join(BASE_DIR, "camera", "face_checkin_single_cam.py")

    success = run_script([
        "python",
        script_path,
        "checkout",
        token
    ])

    return jsonify({
        "status": "OPEN" if success else "DENY"
    })


# ================= RUN =================
if __name__ == "__main__":
    print("AI SERVER RUNNING PORT 5000")

    print("=== ROUTES ===")
    for rule in app.url_map.iter_rules():
        print(rule)

    app.run(host="0.0.0.0", port=5000, threaded=True)