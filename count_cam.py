import cv2

def find_all_cameras():
    """
    Finds and prints the indices of all connected cameras.
    """
    index = 0
    available_cameras = []
    
    while True:
        # Try to open a camera at the current index
        cap = cv2.VideoCapture(index)
        
        # Check if the camera was opened successfully
        if not cap.read()[0]:
            break  # Break the loop if the camera cannot be opened
        else:
            print(f"Camera found at index: {index}")
            available_cameras.append(index)
            cap.release() # Release the camera
            index += 1
            
    return available_cameras

if __name__ == '__main__':
    cameras = find_all_cameras()
    print(f"\nTotal cameras found: {len(cameras)}")
    
    # You can then use one of the found indices in your main script
    if cameras:
        # Example: use the first available camera
        cam_index = cameras[0]
        # self.webcam = cv2.VideoCapture(cam_index)