import cv2
import numpy as np

def run_stereo_viewer(left_cam_index, right_cam_index):
    """
    Captures and displays synchronized video feeds from two cameras and
    computes and displays a depth map.

    Args:
        left_cam_index (int): The device index for the left camera.
        right_cam_index (int): The device index for the right camera.
    """
    # Load stereo calibration parameters
    try:
        with np.load('stereo_calibration.npz') as file:
            mtx_l, dist_l = file['mtx_l'], file['dist_l']
            mtx_r, dist_r = file['mtx_r'], file['dist_r']
            R, T = file['R'], file['T']
            print("Stereo calibration data loaded successfully.")
    except FileNotFoundError:
        print("Error: 'stereo_calibration.npz' not found. Please calibrate your cameras first.")
        return

    # Open the video capture objects for the left and right cameras.
    cap_left = cv2.VideoCapture(left_cam_index)
    cap_right = cv2.VideoCapture(right_cam_index)

    if not cap_left.isOpened():
        print(f"Error: Could not open left camera at index {left_cam_index}")
        return
    if not cap_right.isOpened():
        print(f"Error: Could not open right camera at index {right_cam_index}")
        return

    # # Set both cameras to 1280x720
    # TARGET_W, TARGET_H = 1280, 720
    # cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    # cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    # cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    # cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

    # Give cameras one frame to adjust, then read to confirm actual size
    ret, frame = cap_left.read()
    if not ret:
        print("Failed to read a frame to get dimensions.")
        return
    h, w = frame.shape[:2]

    # Print image dimensions (width, height)
    print(f"Initial frame dimensions: width={w}, height={h}, channels={frame.shape[2] if frame.ndim==3 else 1}")

    # Compute the rectification transforms
    R_l, R_r, P_l, P_r, Q, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, (w, h), R, T)

    # Compute the rectification maps
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R_l, P_l, (w, h), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R_r, P_r, (w, h), cv2.CV_32FC1)

    # Create StereoBM object for disparity calculation
    # These parameters can be tuned for better results.
    # numDisparities: must be a multiple of 16.
    # blockSize: must be an odd number.
    stereo = cv2.StereoBM.create(numDisparities=16*8, blockSize=15)

    print("Stereo viewer with depth map started. Press 'q' to quit.")
    print("Look at the 'Rectified Images' window to check for proper camera calibration.")

    while True:
        # Read a new frame from each camera.
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r:
            print("Failed to grab frames.")
            break

        # Convert to grayscale for disparity calculation
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # Apply the rectification maps to the frames
        rectified_l = cv2.remap(gray_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rectified_r = cv2.remap(gray_r, map1_r, map2_r, cv2.INTER_LINEAR)
        
        # Compute the disparity map
        disparity = stereo.compute(rectified_l, rectified_r).astype(np.float32) / 16.0

        # Normalize the disparity map to be between 0 and 255 for visualization
        normalized_disparity = cv2.normalize(disparity, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply a color map for better visualization
        depth_map = cv2.applyColorMap(normalized_disparity, cv2.COLORMAP_JET)

        # Combine the rectified left and right frames for visual debugging
        rectified_combined = np.concatenate((rectified_l, rectified_r), axis=1)
        # Combine the left camera feed and the depth map
        combined_output = np.concatenate((frame_l, depth_map), axis=1)

        # --- Reduce size for display (change scale as needed) ---
        display_scale = 0.5  # e.g. 0.5 = half size, set to <=1.0

        rectified_disp = cv2.resize(rectified_combined, (0, 0), fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
        frame_l_disp = cv2.resize(frame_l, (0, 0), fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
        depth_map_disp = cv2.resize(depth_map, (frame_l_disp.shape[1], frame_l_disp.shape[0]), interpolation=cv2.INTER_AREA)

        combined_output_disp = np.concatenate((frame_l_disp, depth_map_disp), axis=1)

        # Display the resized frames
        cv2.imshow("Rectified Images", rectified_disp)
        cv2.imshow("Stereo Viewer (Left Camera | Depth Map)", combined_output_disp)

        # Press 'q' to quit the program.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera objects and destroy all windows.
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Call the main function with your camera indices.
    run_stereo_viewer(2, 1)
