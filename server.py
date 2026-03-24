from flask import Flask
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


if __name__ == "__main__":
    print(" Flask API started")
    app.run(host='0.0.0.0', port=5000)