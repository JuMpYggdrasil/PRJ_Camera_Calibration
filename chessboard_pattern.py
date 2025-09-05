import numpy as np
import cv2 as cv
import os

# Define the dimensions of the chessboard in terms of inner corners
# A 9x6 inner corner pattern results in a 10x7 grid of squares.
CHESSBOARD_SIZE = (9, 6)

# Define the size of each square in pixels. A larger value will result in a
# higher resolution image, which is better for printing.
SQUARE_PIXELS = 100

# The total number of squares is (rows+1) x (cols+1)
GRID_ROWS = CHESSBOARD_SIZE[1] + 1
GRID_COLS = CHESSBOARD_SIZE[0] + 1

# Calculate the total image dimensions
width = GRID_COLS * SQUARE_PIXELS
height = GRID_ROWS * SQUARE_PIXELS

# Create a blank white image
chessboard_image = np.full((height, width, 3), 255, dtype=np.uint8)

# Loop through each cell to draw the alternating black squares
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        # A cell is black if the sum of its row and column index is even
        # We start with the top-left square as white.
        if (row + col) % 2 == 1:
            # Calculate the coordinates for the top-left corner of the black square
            start_x = col * SQUARE_PIXELS
            start_y = row * SQUARE_PIXELS
            
            # Draw a filled black rectangle for the square
            cv.rectangle(
                chessboard_image,
                (start_x, start_y),
                (start_x + SQUARE_PIXELS, start_y + SQUARE_PIXELS),
                (0, 0, 0),  # Black color in BGR format
                -1 # -1 fills the rectangle
            )

# Save the generated image
file_path = "printable_chessboard.png"
cv.imwrite(file_path, chessboard_image)

print(f"Generated chessboard pattern saved to '{file_path}'")
print(f"Image dimensions: {width}x{height} pixels")
print("To use this for calibration, ensure that each square on the printed")
print("paper measures exactly 20mm x 20mm or adjust the `SQUARE_SIZE_MM`")
print("variable in the `camera_calibration.py` script accordingly.")
