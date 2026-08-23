import cv2

class CameraStream:
    def __init__(self, camera_id=0):
        # Initialize webcam capture via OpenCV
        self.cap = cv2.VideoCapture(camera_id)

        # Handle camera initialization errors
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open the camera. Please check your connection.")

    def read_frame(self):
        """Reads a single frame from the camera."""
        success, frame = self.cap.read()
        return success, frame
