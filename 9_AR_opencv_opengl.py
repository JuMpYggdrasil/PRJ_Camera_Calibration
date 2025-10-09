from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
import cv2.aruco as aruco
from PIL import Image
import numpy as np
import imutils
import sys

 
from tools.Visualize import draw_axis
from tools.objloader import * #Load obj and corresponding material and textures.
from tools.matrixTrans import extrinsic2ModelView, intrinsic2Project
from tools.Filter import Filter


class AR_render:
    def __init__(self, camera_matrix, dist_coefs, id_to_model, model_scale_dict):
        """[Initialize]
        
        Arguments:
            camera_matrix {[np.array]} -- [your camera intrinsic matrix]
            dist_coefs {[np.array]} -- [your camera difference parameters]
            id_to_model {[dict]} -- [dictionary mapping marker IDs to model paths]
            model_scale {[float]} -- [your model scale size]
        """
        # Initialise webcam and start thread
        src="http://192.168.1.59:43100/videostream.cgi?user=admin&pwd=88888888"
        # self.webcam = cv2.VideoCapture(0)
        self.webcam = cv2.VideoCapture(src)
        self.image_w, self.image_h = map(int, (self.webcam.get(3), self.webcam.get(4)))
        self.initOpengl(self.image_w, self.image_h)
        self.cam_matrix, self.dist_coefs = camera_matrix, dist_coefs
        self.projectMatrix = intrinsic2Project(camera_matrix, self.image_w, self.image_h, 0.01, 100.0)
        self.id_to_model = id_to_model
        self.models = {id: OBJ(path, swapyz=True) for id, path in id_to_model.items()}
        self.model_scale_dict = model_scale_dict
        # Model translate that you can adjust by key board 'w', 's', 'a', 'd'
        self.translate_x, self.translate_y, self.translate_z = 0, 0, 0
        self.pre_extrinsicMatrix = {}
        
        self.filter = Filter()
        

    def loadModel(self, object_path):
        
        """[loadModel from object_path]
        
        Arguments:
            object_path {[string]} -- [path of model]
        """
        self.model = OBJ(object_path, swapyz = True)

  
    def initOpengl(self, width, height, pos_x = 500, pos_y = 500, window_name = b'Aruco Demo'):
        
        """[Init opengl configuration]
        
        Arguments:
            width {[int]} -- [width of opengl viewport]
            height {[int]} -- [height of opengl viewport]
        
        Keyword Arguments:
            pos_x {int} -- [X cordinate of viewport] (default: {500})
            pos_y {int} -- [Y cordinate of viewport] (default: {500})
            window_name {bytes} -- [Window name] (default: {b'Aruco Demo'})
        """
        
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(pos_x, pos_y)
     
        
        
        
        self.window_id = glutCreateWindow(window_name)
        glutDisplayFunc(self.draw_scene)
        glutIdleFunc(self.draw_scene)
        
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glShadeModel(GL_SMOOTH)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        
        # # Assign texture
        glEnable(GL_TEXTURE_2D)
        
        # Add listener
        glutKeyboardFunc(self.keyBoardListener)
        
        # Set ambient lighting
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5,0.5,0.5,1))
        
        
        
        
 
    def draw_scene(self):
        """[Opengl render loop]
        """
        _, image_raw  = self.webcam.read()# get image from webcam camera.
        # --- Undistort the image ---
        image = cv2.undistort(image_raw, self.cam_matrix, self.dist_coefs, None, self.cam_matrix)

        self.draw_background(image)  # draw background
        
        # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.draw_objects(image, mark_size = 0.08) # draw the 3D objects.
        glutSwapBuffers()
    
        
        # TODO add close button
        # key = cv2.waitKey(20)
        
    def draw_axis_opengl(self, axis_length=0.04):
        """
        Draws the X, Y, and Z axes in OpenGL.
        - X-axis is Red
        - Y-axis is Green
        - Z-axis is Blue
        """
        glLineWidth(2.0)
        glBegin(GL_LINES)
        
        # X-axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(axis_length, 0.0, 0.0)

        # Y-axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, axis_length, 0.0)
        
        # Z-axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, axis_length)

        glEnd()
        
        # Explicitly set the color back to white before drawing the model
        glColor3f(1.0, 1.0, 1.0) 
        
        
 
 
 
    def draw_background(self, image):
        """[Draw the background and tranform to opengl format]
        
        Arguments:
            image {[np.array]} -- [frame from your camera]
        """
        # 1. Clear the color and depth buffers. This is done at the beginning of each frame to prepare for drawing.
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        
        # 2. Set up the OpenGL projection and model-view matrices for drawing the 2D background.
        # Setting background image project_matrix and model_matrix.
        # We switch to GL_PROJECTION mode to manipulate the projection matrix.
        glMatrixMode(GL_PROJECTION)
        # Reset the projection matrix to the identity matrix.
        glLoadIdentity()
        # Apply a perspective projection. The arguments are: field of view (33.7), aspect ratio (1.3),
        # near clipping plane (0.1), and far clipping plane (100.0). This creates a 3D view for the background.
        gluPerspective(33.7, 1.3, 0.1, 100.0)
        # Switch to GL_MODELVIEW mode to manipulate the model-view matrix.
        glMatrixMode(GL_MODELVIEW)
        # Reset the model-view matrix to the identity matrix.
        glLoadIdentity()
     
        # 3. Convert the OpenCV image frame to an OpenGL texture format.
        # Flip the image vertically (0) because OpenGL's texture coordinates are inverted relative to OpenCV's.
        bg_image = cv2.flip(image, 0)
        # Convert the NumPy array (from OpenCV) to a PIL Image object.
        bg_image = Image.fromarray(bg_image)
        # Get the width and height of the image.    
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        # Convert the PIL image to a raw byte string with an RGBA format (adding an alpha channel for OpenGL).
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)
  
  
        # 4. Create and bind the OpenGL texture for the background.
        # Generate a single texture ID.
        texid = glGenTextures(1)
        # Bind the generated texture ID as the current 2D texture.
        glBindTexture(GL_TEXTURE_2D, texid)
        # Set the magnification filter to GL_NEAREST for pixelated scaling.
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        # Set the minification filter to GL_NEAREST.
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        # Load the image data into the OpenGL texture. The arguments specify the texture's
        # target, mipmap level (0), internal format (3 components), width, height, border,
        # pixel format (GL_RGBA), data type (GL_UNSIGNED_BYTE), and the image data itself.
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)
        
        # 5. Draw a 2D quad (rectangle) and apply the webcam texture to it.
        # Translate the camera back along the Z-axis so the quad is visible.
        glTranslatef(0.0,0.0,-10.0)
        # Begin drawing a quad primitive.
        glBegin(GL_QUADS)
        # Define the texture coordinates (glTexCoord2f) and vertex positions (glVertex3f) for each corner of the quad.
        # The texture coordinates (0.0 to 1.0) map the texture onto the quad's vertices.
        glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 4.0,  3.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-4.0,  3.0, 0.0)
        # End drawing the quad.
        glEnd()

        # 6. Unbind the texture to prevent it from being accidentally modified by other rendering operations.
        glBindTexture(GL_TEXTURE_2D, 0)
 
 
 
    def draw_objects(self, image, mark_size=0.08):
        """[draw models with opengl]
        
        Arguments:
            image {[np.array]} -- [frame from your camera]
        
        Keyword Arguments:
            mark_size {float} -- [aruco mark size: unit is meter] (default: {0.08})
        """
        # aruco data
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()
        parameters.adaptiveThreshConstant = 7.0

        height, width, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        projectMatrix = intrinsic2Project(self.cam_matrix, width, height, 0.01, 100.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixf(projectMatrix)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        if ids is not None and corners is not None:
            # rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, mark_size, self.cam_matrix, self.dist_coefs)
            zero_dist_coefs = np.zeros(5)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, mark_size, self.cam_matrix, zero_dist_coefs)
            
            # We'll use the first detected marker (e.g., ID 0) as the origin.
            zero_dist_coefs = np.zeros(5) # Reuse or redefine here for clarity
            if 0 in ids:
                i = np.where(ids.flatten() == 0)[0][0]
                rvec = rvecs[i]
                tvec = tvecs[i]
                
                draw_axis(image, rvec, tvec, self.cam_matrix, zero_dist_coefs)# draw on undistorted image
                # draw_axis(image, rvec, tvec, self.cam_matrix, self.dist_coefs)# draw on distorted image
                
                if self.filter.update(tvec):
                    model_matrix = extrinsic2ModelView(rvec, tvec)
                    self.pre_extrinsicMatrix[0] = model_matrix
                else:
                    model_matrix = self.pre_extrinsicMatrix.get(0)

                if model_matrix is not None:
                    glLoadMatrixf(model_matrix)
                    
                    
                    
                    # --- Draw the purple dot at the World Coordinate Frame origin ---
                    glPointSize(10.0) # Set the size of the point
                    glColor3f(1.0, 0.0, 1.0) # Set the color to purple (R, G, B)
                    glBegin(GL_POINTS)
                    glVertex3f(0.0, 0.0, 0.0) # Draw a point at the origin of the World Frame
                    glEnd()
                    # --- End of new code ---
                    
                    # Explicitly set the color back to white before drawing the model
                    glColor3f(1.0, 1.0, 1.0) 
                    
            # Store the 3D positions of the markers we care about.
            marker_positions = {}
            
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.models:
                    rvec = rvecs[i]
                    tvec = tvecs[i]
                    draw_axis(image, rvec, tvec, self.cam_matrix, zero_dist_coefs)# draw on undistorted image
                    # draw_axis(image, rvec, tvec, self.cam_matrix, self.dist_coefs)# draw on distorted image
                    
                    if self.filter.update(tvec):
                        model_matrix = extrinsic2ModelView(rvec, tvec)
                        self.pre_extrinsicMatrix[marker_id] = model_matrix
                    else:
                        model_matrix = self.pre_extrinsicMatrix.get(marker_id)
                    if model_matrix is not None:
                        glLoadMatrixf(model_matrix) # sets the coordinate system for the next drawing call
                        # --- Draw the OpenGL axis ---
                        self.draw_axis_opengl(axis_length=0.05)
                        
                        scale = self.model_scale_dict.get(marker_id, 0.01)  # Default scale if not found
                        glScaled(scale, scale, scale)
                        glTranslatef(self.translate_x, self.translate_y, self.translate_z)
                        glCallList(self.models[marker_id].gl_list)

                    # Store the 3D position of the detected marker.
                    marker_positions[marker_id] = tvecs[i][0]
            
            
            
            
            # --- distance calculation and drawing ---
            # Check if both markers 0 and 1 are detected.
            if 0 in marker_positions and 1 in marker_positions:
                point1_3d = marker_positions[0]
                point2_3d = marker_positions[1]

                # Calculate the Euclidean distance in 3D.
                distance = np.linalg.norm(point1_3d - point2_3d)

                # Project the 3D points to 2D image coordinates to draw the line.
                # We use the midpoint for the text label.
                mid_point_3d = (point1_3d + point2_3d) / 2
                
                # Since the points are already relative to the camera,
                # we can pass zero vectors for rvec and tvec.
                r_vec_zero = np.zeros((3, 1))
                t_vec_zero = np.zeros((3, 1))
                
                # points_2d, _ = cv2.projectPoints(
                #     np.array([point1_3d, point2_3d, mid_point_3d]), 
                #     r_vec_zero, 
                #     t_vec_zero, 
                #     self.cam_matrix, 
                #     self.dist_coefs
                # )
                points_2d, _ = cv2.projectPoints(
                    np.array([point1_3d, point2_3d, mid_point_3d]), 
                    r_vec_zero, 
                    t_vec_zero, 
                    self.cam_matrix, 
                    zero_dist_coefs
                )
                

                # Extract the 2D points.
                point1_2d = tuple(points_2d[0][0].astype(int))
                point2_2d = tuple(points_2d[1][0].astype(int))
                mid_point_2d = tuple(points_2d[2][0].astype(int))

                # Draw a line between the two marker centers.
                cv2.line(image, point1_2d, point2_2d, (0, 255, 0), 2)

                # Format the distance text.
                distance_text = f"Distance: {distance:.2f} m"

                # Draw the distance text at the midpoint.
                cv2.putText(
                    image, 
                    distance_text, 
                    mid_point_2d, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2, 
                    cv2.LINE_AA
                )
                
        cv2.imshow("Frame", image)
        cv2.waitKey(20)

    def keyBoardListener(self, key, x, y):
        """[Use key board to adjust model size and position]
        
        Arguments:
            key {[byte]} -- [key value]
            x {[x cordinate]} -- []
            y {[y cordinate]} -- []
        """
        key = key.decode('utf-8')
        if key == '=':
            self.model_scale += 0.01
        elif key == '-':
            self.model_scale -= 0.01
        elif key == 'w':
            self.translate_x -= 0.1
        elif key == 's':
            self.translate_x += 0.1
        elif key == 'a':
            self.translate_y -= 0.1
        elif key == 'd':
            self.translate_y += 0.1
             
        
    def run(self):
        # Begin to render
        glutMainLoop()
  

if __name__ == "__main__":
    # The value of cam_matrix and dist_coeff from your calibration by using chessboard.
    
    try:
        with np.load('calibration_results.npz') as file:
            cam_matrix = file['camera_matrix']
            dist_coeff = file['dist_coeffs']
            print("Calibration data loaded successfully.")
    except FileNotFoundError:
        print("Error: 'calibration_results.npz' not found.")
        cam_matrix = np.array([
            [963.4519793109993, 0, 647.0863663141905],
            [0, 966.0565298361108, 352.23753011981177],
            [0, 0, 1]
        ])
  
        dist_coeff = np.array([-0.15259701966137876, 0.6092617145206677, 0.0007901395004658092, 0.0026990411152102638, -0.6577414700462231]) 
    # Map marker IDs to model paths
    id_to_model = {
        0: './Models/Barn/ban.obj',
        1: './Models/Monster/Sinbad_4_000001.obj'
    }
    model_scale_dict = {
        0: 0.01,  # scale for marker 0
        1: 0.03   # scale for marker 1
    }
    ar_instance = AR_render(cam_matrix, dist_coeff, id_to_model, model_scale_dict)
    ar_instance.run()