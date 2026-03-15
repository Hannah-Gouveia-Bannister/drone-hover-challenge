import cv2

def find_webcam_port():
    for port in range(5):
        cap = cv2.VideoCapture(port)
        if cap.isOpened():
            print(f"Webcam found on port: {port}")
            ret, frame = cap.read()
            if ret:
                print(f"  → Successfully captured frame from port {port}")
            cap.release()
        else:
            print(f"Port {port}: No camera found")

find_webcam_port()