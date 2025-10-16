import cv2
import datetime
import os

def take_stereo_photos(left_cam_index, right_cam_index):
    """
    Captures synchronized photos from two cameras when the spacebar is pressed.
    
    Args:
        left_cam_index (int): The device index for the left camera.
        right_cam_index (int): The device index for the right camera.
    """
    # Open both cameras
    cap_left = cv2.VideoCapture(left_cam_index)
    cap_right = cv2.VideoCapture(right_cam_index)
    
    # Set resolutions
    # Camera at index 0 (left) will have a resolution of 1280x960
    # Camera at index 2 (right) will have a resolution of 640x480
    high_res_w, high_res_h = 640, 480
    low_res_w, low_res_h = 640, 480
    
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, high_res_w)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, high_res_h)
    
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, low_res_w)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, low_res_h)

    # Check if cameras are opened correctly
    if not cap_left.isOpened():
        print(f"Error: Could not open camera at index {left_cam_index}")
        return
    if not cap_right.isOpened():
        print(f"Error: Could not open camera at index {right_cam_index}")
        return

    # Create directories to save photos
    os.makedirs('left_images', exist_ok=True)
    os.makedirs('right_images', exist_ok=True)
    print("Press SPACEBAR to take photos. Press 'q' to quit.")

    while True:
        # Read frames from both cameras
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        # Check if frames were read correctly
        if not ret_left or not ret_right:
            print("Failed to grab frames from one or both cameras. Exiting...")
            break

        # Display frames
        cv2.imshow("Left Camera", frame_left)
        cv2.imshow("Right Camera", frame_right)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # If the spacebar is pressed, take a photo
        if key == ord(' '):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
            left_filename = os.path.join('left_images', f'photo_{timestamp}.jpg')
            right_filename = os.path.join('right_images', f'photo_{timestamp}.jpg')
            
            # Save the photos
            cv2.imwrite(left_filename, frame_left)
            cv2.imwrite(right_filename, frame_right)
            
            print(f"Saved photos: {left_filename} and {right_filename}")

        # If 'q' is pressed, quit the loop
        elif key == ord('q'):
            break

    # Release cameras and close all windows
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Call the function with the desired camera indices
    # Now using cameras at indices 0 and 1.
    take_stereo_photos(2, 1)
