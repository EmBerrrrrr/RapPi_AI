import cv2

for i in range(6):
    print(f"\nTesting camera {i}")
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    ret, frame = cap.read()

    if ret:
        print(f"✅ Camera {i} WORKS")
        cv2.imshow(f"Cam {i}", frame)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
    else:
        print(f"❌ Camera {i} FAKE / NOT WORKING")

    cap.release()