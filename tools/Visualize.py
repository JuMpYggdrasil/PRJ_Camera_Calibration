# import cv2
# import numpy as np

# axis_axis = np.float32([[0,0,0], [0.03,0,0],[0,0.03,0],[0,0,0.03]]).reshape(-1,3)
# def draw_axis(img, rvecs, tvecs, mtx, dist):
#     imgpts, jac = cv2.projectPoints(axis_axis, rvecs, tvecs, mtx, dist)
#     oriPoint = tuple(imgpts[0].ravel())
#     x_axis = tuple(imgpts[1].ravel())
#     y_axis = tuple(imgpts[2].ravel())
#     z_axis = tuple(imgpts[3].ravel())
    
#     img = cv2.line(img, oriPoint, x_axis, (255, 0, 0), 5)
#     img = cv2.line(img, oriPoint, y_axis, (0, 255, 255), 5)
#     img = cv2.line(img, oriPoint, z_axis, (0, 0, 255), 5)

#     return img

import cv2
import numpy as np

def draw_axis(img, rvec, tvec, K, D):
    # This is an example of what your draw_axis function might look like.
    # The key is to convert the points to integers before drawing.
    
    # Define your axis points in 3D space
    axis_points = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0]]).reshape(-1, 3)
    
    # Project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis_points, rvec, tvec, K, D)
    
    # The imgpts will contain floats, which is what's causing the error.
    # Extract the origin and axis points, then convert them to integers.
    ori = tuple(np.int32(imgpts[3].ravel()))
    x_axis = tuple(np.int32(imgpts[0].ravel()))
    y_axis = tuple(np.int32(imgpts[1].ravel()))
    z_axis = tuple(np.int32(imgpts[2].ravel()))

    # Now, draw the lines using the integer points
    img = cv2.line(img, ori, x_axis, (255, 0, 0), 5) # Red line for X-axis
    img = cv2.line(img, ori, y_axis, (0, 255, 0), 5) # Green line for Y-axis
    img = cv2.line(img, ori, z_axis, (0, 0, 255), 5) # Blue line for Z-axis
    
    return img