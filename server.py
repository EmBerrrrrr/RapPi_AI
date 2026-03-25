from flask import Flask, request
import subprocess

app = Flask(__name__)

def run_script(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120 
        )

        print("\n===== SCRIPT LOG =====")
        print(result.stdout)
        print(result.stderr)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT")
        return False

    except Exception as e:
        print("❌ ERROR:", e)
        return False

@app.route('/checkin')
def checkin():
    if run_script(["python", "camera/checkin_capture.py"]):
        return "OPEN"
    return "DENY"


@app.route('/checkout')
def checkout():
    if run_script(["python", "camera/checkout_capture.py"]):
        return "OPEN"
    return "DENY"

@app.route('/face_checkin', methods=['POST'])
def face_checkin():
    token = request.json.get("token", "")

    print(f"[FACE CHECKIN] Token: {token}")

    if run_script([
        "python",
        "camera/face_checkin_single_cam.py",
        "checkin",
        token
    ]):
        return "OPEN"

    return "DENY"


@app.route('/face_checkout', methods=['POST'])
def face_checkout():
    token = request.json.get("token", "")

    print(f"[FACE CHECKOUT] Token: {token}")

    if run_script([
        "python",
        "camera/face_checkin_single_cam.py",
        "checkout",
        token
    ]):
        return "OPEN"

    return "DENY"

if __name__ == "__main__":
    print("AI Flask running at port 5000")
    app.run(host='0.0.0.0', port=5000)