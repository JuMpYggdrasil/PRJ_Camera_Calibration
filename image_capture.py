import cv2 as cv
import time
import os

# --- Configuration ---
# Define the directory where the captured photos will be saved.
output_dir = "captured_photos"

# Define the desired width and height for the video stream and captured images.
# These values can be changed to match your preferred resolution.
frame_width = 1280
frame_height = 720

# Define the maximum number of photos to capture.
max_photos = 30

# --- Setup ---
# Create the output directory if it doesn't already exist.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Initialize the video capture object.
# The argument `0` refers to the default webcam. If you have multiple,
# you might need to change this to 1, 2, etc.
cap = cv.VideoCapture(0)

# Set the resolution of the video stream.
cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

# Check if the webcam was opened successfully.
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully.")
print("Press 'q' to quit.")
print("Press 'Spacebar' to capture a photo.")

# Initialize the photo counter.
photo_count = 0

# --- Main Loop ---
while True:
    # Read a frame from the webcam.
    ret, frame = cap.read()
    
    # Check if the frame was read successfully.
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break

    # Display the live video feed.
    cv.imshow('Webcam Live Feed', frame)

    # Check for a key press.
    key = cv.waitKey(1) & 0xFF
    
    # If the user presses the 'spacebar'
    if key == ord(' '):
        # --- Capture the photo instantly ---
        # Generate a filename with a timestamp.
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(output_dir, f"webcam_photo_{timestamp}.jpg")
        
        # Save the frame as a JPEG image.
        cv.imwrite(filename, frame)
        photo_count += 1
        print(f"Photo {photo_count} of {max_photos} captured and saved as: {filename}")
        
    # Exit the loop if the maximum number of photos has been reached or 'q' is pressed.
    if key == ord('q') or photo_count >= max_photos:
        print("Maximum number of photos captured. Stopping...")
        break

# --- Cleanup ---
# Release the video capture object and close all OpenCV windows.
cap.release()
cv.destroyAllWindows()
print("Webcam released and all windows closed.")
