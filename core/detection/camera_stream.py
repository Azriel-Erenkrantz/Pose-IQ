import cv2
import time

class CameraStream:
    def __init__(self, camera_id=0):
        # Initialize webcam capture via OpenCV
        self.cap = cv2.VideoCapture(camera_id)
        
        # Handle camera initialization errors
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open the camera. Please check your connection.")
            
        # Variables for FPS calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0

    def read_frame(self):
        """Reads a single frame from the camera."""
        success, frame = self.cap.read()
        return success, frame

    def calculate_fps(self):
        """Measures and calculates the Frames Per Second (FPS)."""
        self.new_frame_time = time.time()
        # Avoid division by zero on the first frame
        if self.prev_frame_time == 0:
            fps = 0
        else:
            fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        
        return int(fps)

    def run(self):
        """Standalone test loop for the camera (Used mostly for debugging)."""
        print("Starting camera stream... Press 'q' to exit.")
        
        while True:
            success, frame = self.read_frame()
            if not success:
                print("Failed to read from camera. Exiting...")
                break

            fps = self.calculate_fps()
            cv2.putText(frame, f"FPS: {fps}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Camera Stream', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        stream = CameraStream()
        stream.run()
    except Exception as e:
        print(f"Camera error: {e}")