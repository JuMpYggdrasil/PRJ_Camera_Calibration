import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all images.
objpoints = [] # 3d point in real world space
imgpoints_l = [] # 2d points in image plane.
imgpoints_r = [] # 2d points in image plane.

images_l = sorted(glob.glob('left_images/*.jpg'))
images_r = sorted(glob.glob('right_images/*.jpg'))

# Ensure you have the same number of images for both cameras
if len(images_l) != len(images_r):
    print("Error: The number of images for left and right cameras do not match.")
    exit()
    
# Get image size from the first image before the loop
img_l = cv2.imread(images_l[0])
gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
h, w = gray_l.shape[:2]

print("Starting chessboard corner detection...")

for i, (fname_l, fname_r) in enumerate(zip(images_l, images_r)):
    img_l = cv2.imread(fname_l)
    img_r = cv2.imread(fname_r)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9,6), None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9,6), None)

    if ret_l and ret_r:
        objpoints.append(objp)
        
        # Refine corners for subpixel accuracy
        corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
        imgpoints_l.append(corners2_l)

        corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)
        imgpoints_r.append(corners2_r)

        # Draw and display the corners (optional)
        img_l = cv2.drawChessboardCorners(img_l, (9,6), corners2_l, ret_l)
        img_r = cv2.drawChessboardCorners(img_r, (9,6), corners2_r, ret_r)
        
        cv2.imshow('Left Chessboard', img_l)
        cv2.imshow('Right Chessboard', img_r)
        cv2.waitKey(100)
    else:
        print(f"Corners not found in image pair {i+1}")

cv2.destroyAllWindows()

# Get image size
h, w = gray_l.shape[:2]

# --- Monocular Calibration (Step 4) ---
print("Calibrating left camera...")
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, (w,h), None, None)

print("Calibrating right camera...")
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, (w,h), None, None)

# --- Stereo Calibration (Step 5) ---
print("Performing stereo calibration...")
ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, (w,h), criteria, cv2.CALIB_FIX_INTRINSIC
)

# You can save the calibration results to a file
np.savez('stereo_calibration.npz', mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r, R=R, T=T, E=E, F=F)

print("Stereo Calibration Complete.")
print("Left Camera Matrix:\n", mtx_l)
print("\nRight Camera Matrix:\n", mtx_r)
print("\nRotation Matrix (R):\n", R)
print("\nTranslation Vector (T):\n", T)