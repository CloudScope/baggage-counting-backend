import cv2

def access_cctv_camera(stream_url):
    """
    Access and display CCTV camera stream using OpenCV.

    Args:
        stream_url (str): RTSP or HTTP stream URL of the CCTV camera.
    """
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("❌ Unable to connect to the camera.")
        return

    print("✅ Camera stream started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to retrieve frame.")
            break

        cv2.imshow('CCTV Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    # Example RTSP URL (replace with your actual camera stream URL)
    rtsp_url = "rtsp://admin:admin%404321@117.242.184.12:554/cam/realmonitor?channel=1&subtype=0"
    access_cctv_camera(rtsp_url)