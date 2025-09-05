import numpy as np
import cv2 as cv
import glob

# --- 1. Define Chessboard Parameters ---
# Number of inner corners on the chessboard.
# A standard 9x6 board has 8x5 inner corners.
chessboard_size = (9, 6)

# Size of a single square in meters (or any unit of your choice).
# This is used to calculate the real-world 3D coordinates of the corners.
square_size = 0.0265  # 26.5 mm

# --- 2. Setup Object and Image Points ---
# Object points are the 3D coordinates of the chessboard corners in the real world.
# We assume the chessboard is on the Z=0 plane for simplicity.
# This array will be the same for all images.
object_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
object_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store the 3D object points and 2D image points from all calibration images.
all_object_points = []  # 3D points in the real world
all_image_points = []  # 2D points in the image plane

# --- 3. Find Chessboard Corners in Images ---
# Get the list of all image files in the 'calibration_images' folder.
images = glob.glob('captured_photos/*.jpg')

# Check if any images were found.
if not images:
    print("Error: No images found in the 'calibration_images' directory.")
    print("Please add your chessboard photos to this folder and try again.")
    exit()

# Loop through each image to find the chessboard corners.
for i, filename in enumerate(images):
    print(f"Processing image {i+1}/{len(images)}: {filename}")
    
    # Read the image in grayscale for corner detection.
    img = cv.imread(filename)
    if img is None:
        print(f"Warning: Could not read image {filename}. Skipping.")
        continue
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Find the chessboard corners.
    # The `cv.CALIB_CB_ADAPTIVE_THRESH` flag improves corner detection in varying lighting conditions.
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    
    # If a full set of corners was found, refine them and add to our lists.
    if ret:
        # Refine the corners to sub-pixel accuracy.
        # This increases the accuracy of the calibration.
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Add the points to our lists.
        all_object_points.append(object_points)
        all_image_points.append(corners_refined)
        
        # Optionally, draw the found corners on the image to visualize the detection.
        cv.drawChessboardCorners(img, chessboard_size, corners_refined, ret)
        cv.imshow('Corners Found', img)
        cv.waitKey(500) # Wait for 500 milliseconds
    else:
        print(f"Warning: Chessboard corners not found in {filename}. Skipping this image.")

# Close the visualization window after the loop.
cv.destroyAllWindows()

# --- 4. Calibrate the Camera ---
# Perform the calibration using the collected points.
# This function returns the camera matrix, distortion coefficients,
# rotation vectors, and translation vectors.
print("\nStarting camera calibration...")
if len(all_object_points) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        all_object_points, all_image_points, gray.shape[::-1], None, None)
else:
    print("Error: No valid chessboard images were found. Calibration cannot be performed.")
    exit()

# --- 5. Save the Results ---
# Save the camera matrix and distortion coefficients to a file.
# The `calibration_results.npz` file can then be loaded by other scripts.
np.savez('calibration_results.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print("\nCalibration successful!")
print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)
print("\nCalibration results saved to 'calibration_results.npz'. You can now use this file for image and video undistortion.")

# --- 6. Optional: Undistort a sample image for verification ---
# Load a new image to verify the calibration.
sample_image_path = images[0]
sample_img = cv.imread(sample_image_path)
h, w = sample_img.shape[:2]

# Get the optimal camera matrix for undistortion.
new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# Undistort the image.
undistorted_img = cv.undistort(sample_img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# Crop the image to the valid ROI to remove black borders.
x, y, w, h = roi
undistorted_img_cropped = undistorted_img[y:y+h, x:x+w]

# Display both the original and undistorted images.
cv.imshow('Original Image', sample_img)
cv.imshow('Undistorted Image', undistorted_img_cropped)
cv.waitKey(0)

# Final cleanup
cv.destroyAllWindows()
