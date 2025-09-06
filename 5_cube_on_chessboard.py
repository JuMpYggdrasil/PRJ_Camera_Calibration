import cv2 as cv
import numpy as np

# --- 1. Load Camera Calibration Data ---
try:
    with np.load('calibration_results.npz') as file:
        camera_matrix = file['camera_matrix']
        dist_coeffs = file['dist_coeffs']
    print("Calibration data loaded successfully.")
except FileNotFoundError:
    print("Error: 'calibration_results.npz' not found.")
    print("Please run the camera_calibration.py script first to generate this file.")
    exit()

# --- 2. Define Chessboard and 3D Model Points ---
# Chessboard inner corners
chessboard_size = (9, 6)

# The real-world 3D coordinates of the chessboard corners.
# We assume the chessboard is on the Z=0 plane.
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Define the 3D points of the simple model (a cube) to be rendered.
# Each point is relative to the origin of the chessboard, which is the top-left corner.
# The size is based on one chessboard square.
model_points = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                           [0,0,-1], [0,1,-1], [1,1,-1], [1,0,-1]])

# --- 3. Initialize Video Capture ---
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set a fixed resolution for consistent processing
frame_width = 1280
frame_height = 720
cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

# --- 4. Main Loop for Rendering ---
print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Find the chessboard corners in the current frame.
    ret_corners, corners = cv.findChessboardCorners(gray, chessboard_size, None)

    # If the corners are found, proceed with pose estimation and rendering.
    if ret_corners:
        # Get the rotation and translation vectors using solvePnP.
        # This function estimates the pose of the chessboard relative to the camera.
        ret_solvepnp, rvec, tvec = cv.solvePnP(objp, corners, camera_matrix, dist_coeffs)

        # Project the 3D model points onto the 2D image plane.
        # This transforms our 3D cube coordinates into 2D pixel coordinates.
        image_points, _ = cv.projectPoints(model_points, rvec, tvec, camera_matrix, dist_coeffs)
        image_points = np.int32(image_points).reshape(-1, 2)

        # Draw the 3D model (cube) on the frame.
        # Draw the base (front face) of the cube.
        frame = cv.drawContours(frame, [image_points[:4]],-1,(0,255,0),3)
        # Draw the top (back face) of the cube.
        frame = cv.drawContours(frame, [image_points[4:]],-1,(0,0,255),3)
        # Draw the connecting lines between the front and back faces.
        for i, j in zip(range(4), range(4,8)):
            cv.line(frame, tuple(image_points[i]), tuple(image_points[j]), (255,0,0), 3)

    # Display the final frame with the rendered model.
    cv.imshow('3D Model on Chessboard', frame)

    # Break the loop if 'q' is pressed.
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. Cleanup ---
cap.release()
cv.destroyAllWindows()
print("Webcam released and all windows closed.")
