import cv2 as cv
import numpy as np
from stl import mesh

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

# --- 2. Load 3D Model from STL file ---
try:
    # Load the STL file. Make sure your STL file is in the same directory as this script.
    model_mesh = mesh.Mesh.from_file('model.stl')
    
    # Extract vertices from the STL mesh
    model_vertices = model_mesh.vectors.reshape(-1, 3)
    
    # Normalize the model's coordinates to fit the chessboard square size.
    # We will scale the STL model to fit within a similar unit square.
    max_coords = np.max(model_vertices, axis=0)
    min_coords = np.min(model_vertices, axis=0)
    model_size = max_coords - min_coords
    scale = 3.0 / max(model_size)
    
    # Center the model's origin to the center of its base and apply scaling.
    model_vertices = (model_vertices - min_coords - model_size/2) * scale
    
    # Create the transformed faces array for drawing.
    # We use the scaled vertices to create the faces for rendering.
    model_faces = model_vertices.reshape(-1, 3, 3)
    
    print("3D model loaded and prepared.")

except FileNotFoundError:
    print("Error: 'model.stl' not found.")
    print("Please place an STL file named 'model.stl' in the same directory as this script.")
    exit()

# --- 3. Define Chessboard Points ---
chessboard_size = (9, 6)
# The real-world 3D coordinates of the chessboard inner corners.
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# --- 4. Initialize Video Capture ---
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
# Set a fixed resolution for consistent processing
frame_width = 1280
frame_height = 720
cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

# --- 5. Main Loop for Rendering ---
print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Find the chessboard corners in the current frame.
    ret_corners, corners = cv.findChessboardCorners(gray, chessboard_size, None)
    
    if ret_corners:
        # Get the rotation and translation vectors using solvePnP.
        ret_solvepnp, rvec, tvec = cv.solvePnP(objp, corners, camera_matrix, dist_coeffs)
        
        # Project all the model's vertices onto the 2D image plane.
        image_points, _ = cv.projectPoints(model_vertices, rvec, tvec, camera_matrix, dist_coeffs)
        image_points = np.int32(image_points).reshape(-1, 2)
        
        # Draw the 3D model (faces) on the frame.
        for face in model_faces:
            # The projected face points are taken directly from the projected vertices.
            projected_face = np.array([
                image_points[np.where(np.all(model_vertices == face[0], axis=1))[0][0]],
                image_points[np.where(np.all(model_vertices == face[1], axis=1))[0][0]],
                image_points[np.where(np.all(model_vertices == face[2], axis=1))[0][0]]
            ])
            cv.fillPoly(frame, [projected_face], (0, 255, 0))
            
    # Display the final frame with the rendered model.
    cv.imshow('3D Model on Chessboard', frame)
    
    # Break the loop if 'q' is pressed.
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. Cleanup ---
cap.release()
cv.destroyAllWindows()
print("Webcam released and all windows closed.")
