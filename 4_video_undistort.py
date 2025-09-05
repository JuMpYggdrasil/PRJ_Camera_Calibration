import cv2 as cv
import numpy as np

# --- 1. Load Calibration Data ---
try:
    with np.load('calibration_results.npz') as file:
        camera_matrix = file['camera_matrix']
        dist_coeffs = file['dist_coeffs']
        print("Calibration data loaded successfully.")
except FileNotFoundError:
    print("Error: 'calibration_results.npz' not found.")
    print("Please run the camera_calibration.py script first to generate this file.")
    exit()

# --- 2. Initialize Video Capture ---
# Use the default webcam. Change the index if you have multiple cameras.
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set a fixed resolution for the video stream for consistent processing.
# You can change these values as needed.
frame_width = 1280
frame_height = 720
cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

# Get the new optimal camera matrix and a region of interest.
# The optimal matrix is the best fit for undistorting the images.
new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (frame_width, frame_height), 1, (frame_width, frame_height))
x, y, w, h = roi
print(f"Optimal Camera Matrix created. ROI (x, y, w, h): ({x}, {y}, {w}, {h})")

print("Press 'q' to quit the application.")

# --- 3. Main Loop for Frame Processing ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break
    
    # a) Undistort the frame
    undistorted_frame = cv.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # b) Crop the region of interest from the undistorted frame
    cropped_frame = undistorted_frame[y:y+h, x:x+w]

    # Display both the original and the undistorted/cropped frames for comparison
    cv.imshow('Original Frame', frame)
    cv.imshow('Undistorted & Cropped Frame', cropped_frame)
    
    # Break the loop if the 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. Cleanup ---
cap.release()
cv.destroyAllWindows()
print("Webcam released and all windows closed.")
