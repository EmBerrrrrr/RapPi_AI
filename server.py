from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route('/checkin')
def checkin():
    try:
        result = subprocess.run(
            ["python", "camera/checkin_capture.py"],
            capture_output=True,
            text=True
        )

        print("\n===== CHECK-IN LOG =====")
        print(result.stdout)
        print(result.stderr)

        if result.returncode == 0:
            return "OPEN"
        else:
            return "DENY"

    except Exception as e:
        print(" CHECK-IN ERROR:", e)
        return "DENY"


@app.route('/checkout')
def checkout():
    try:
        result = subprocess.run(
            ["python", "camera/checkout_capture.py"],
            capture_output=True,
            text=True
        )

        print("\n===== CHECK-OUT LOG =====")
        print(result.stdout)
        print(result.stderr)

        if result.returncode == 0:
            return "OPEN"
        else:
            return "DENY"

    except Exception as e:
        print("CHECK-OUT ERROR:", e)
        return "DENY"

@app.route('/face_checkin', methods=['POST'])
def face_checkin():
    try:
        token = request.json.get("token", "")

        result = subprocess.run(
            ["python", "face_checkin_single_cam.py", "checkin", token],
            capture_output=True,
            text=True
        )

        print("\n===== FACE CHECKIN LOG =====")
        print(result.stdout)
        print(result.stderr)

        if result.returncode == 0:
            return "OPEN"
        else:
            return "DENY"

    except Exception as e:
        print("FACE CHECKIN ERROR:", e)
        return "DENY"


@app.route('/face_checkout', methods=['POST'])
def face_checkout():
    try:
        token = request.json.get("token", "")

        result = subprocess.run(
            ["python", "face_checkin_single_cam.py", "checkout", token],
            capture_output=True,
            text=True
        )

        print("\n===== FACE CHECKOUT LOG =====")
        print(result.stdout)
        print(result.stderr)

        if result.returncode == 0:
            return "OPEN"
        else:
            return "DENY"

    except Exception as e:
        print("FACE CHECKOUT ERROR:", e)
        return "DENY"

if __name__ == "__main__":
    print("Flask API started")
    app.run(host='0.0.0.0', port=5000)