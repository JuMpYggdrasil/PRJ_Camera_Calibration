import numpy as np
import cv2 as cv

# --- 1. Main Loop for Detection ---
def main():
    # Initialize webcam with OpenCV
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    frame_width = 1280
    frame_height = 720
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    # --- ArUco Setup ---
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    aruco_params = cv.aruco.DetectorParameters()
    aruco_detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # --- Load Camera Calibration from File ---
    try:
        with np.load('calibration_results.npz') as data:
            camera_matrix = data['camera_matrix']
            dist_coeffs = data['dist_coeffs']
        print("Camera calibration parameters loaded successfully.")
    except FileNotFoundError:
        print("Error: 'calibration_results.npz' not found.")
        print("Please run the 'camera_calibration.py' script first to generate this file.")
        return

    # This is the real-world size of your printed marker in meters.
    # You MUST measure your printed marker and change this value.
    marker_size = 0.05 # Example: 5cm
    
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break
        
        # Convert to grayscale for detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Detect the markers in the frame
        corners, ids, rejected = aruco_detector.detectMarkers(gray)
        
        # --- Find and Render Marker ---
        if ids is not None:
            # Draw the detected markers on the frame
            cv.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Estimate pose of each marker
            rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

            # Draw coordinate axes for each marker
            for i in range(len(ids)):
                cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)
            
        # Display the resulting frame
        cv.imshow('ArUco Marker Detection', frame)
        
        # Quit when 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv.destroyAllWindows()
        
if __name__ == '__main__':
    main()
