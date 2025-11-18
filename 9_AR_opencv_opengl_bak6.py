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
        # src="http://192.168.1.59:43100/videostream.cgi?user=admin&pwd=88888888"
        src="http://192.168.1.59:45629/videostream.cgi?user=admin&pwd=88888888"
        self.webcam = cv2.VideoCapture(0)
        # self.webcam = cv2.VideoCapture(src)
        print("webcam name: ", self.webcam.getBackendName())
        self.image_w, self.image_h = map(int, (self.webcam.get(3), self.webcam.get(4)))
        print("wh: ",self.image_w, self.image_h)
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
        
        """[Init opengl configuration]"""
        
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
        glEnable(GL_DEPTH_TEST) # Enables depth for proper 3D rendering
        
        # --- NEW/UPDATED LIGHTING SETUP FOR SHADING ---
        glEnable(GL_LIGHTING)    # 1. Enable the global lighting system
        glEnable(GL_LIGHT0)      # 2. Enable Light Source 0

        # allow glColor to affect material when lighting is enabled
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # normalize normals after scaling (important for correct lighting)
        glEnable(GL_NORMALIZE)

        # 3. Define the light source properties
        # Simple white directional light coming from (+X, +Y, +Z)
        light_ambient = (0.4, 0.4, 0.4, 1.0) # Low ambient light to give some color to unlit areas
        light_diffuse = (1.0, 1.0, 1.0, 1.0) # Bright white diffuse light
        light_specular = (1.0, 1.0, 1.0, 1.0) # Bright specular highlights
        light_position = (0.0, 1.0, 1.0, 0.0) # Directional light (W=0.0)
        
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)

        # ---------------------------------------------
        
        # # Assign texture (keep as is)
        glEnable(GL_TEXTURE_2D)
        
        # Add listener (keep as is)
        glutKeyboardFunc(self.keyBoardListener)
        
        
 
    def draw_scene(self):
        """[Opengl render loop]
        """
        _, image_raw  = self.webcam.read()# get image from webcam camera.
        # --- Undistort the image ---
        image = cv2.undistort(image_raw, self.cam_matrix, self.dist_coefs, None, self.cam_matrix)

        # Clear buffer ONCE
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.draw_background(image)  # Draw background first

        # 5️⃣ Setup camera projection for AR objects
        height, width = image.shape[:2]
        projectMatrix = intrinsic2Project(self.cam_matrix, width, height, 0.01, 500.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixf(projectMatrix)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.draw_objects(image, mark_size=0.17) # draw the 3D objects.(marker size)
        
        glutSwapBuffers()
    
        
        # TODO add close button
        # key = cv2.waitKey(20)
        
    def draw_axis_opengl(self, axis_length=0.1):
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
        
    def draw_tube_opengl(self, marker_id_from,marker_id_to,n_tube=1,radius = 0.03, axis_length=1):
        """
        Draws the X, Y, and Z axes in OpenGL.
        - X-axis is Red
        - Y-axis is Green
        - Z-axis is Blue
        """
        model_matrix = self.pre_extrinsicMatrix.get(marker_id_from)#marker_id
        
        if model_matrix is not None:
            glLoadMatrixf(model_matrix)  # sets the coordinate system for the next drawing call

            # --- Draw Y-axis line ---
            glLineWidth(2.0)
            glPushMatrix()
            glTranslatef(0.0, 0.0, 0.1)  # small Z-offset if desired
            glBegin(GL_LINES)
            glColor3f(0.0, 0.0, 0.0)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(0.0, axis_length, 0.0)
            glEnd()
            glPopMatrix()

            # --- Setup for Cylinders ---
            # Set the color for all subsequent geometry to Black (R=0.0, G=0.0, B=0.0)
            glColor3f(0.0, 0.0, 0.0)

            # Create the GLU quadric object once outside the loop
            quad = gluNewQuadric()
            gluQuadricNormals(quad, GLU_SMOOTH)

            # Define parameters
            num_cylinders = n_tube
            x_step = 0.1
            z_offset = 0.1
            radius = 0.03
            length = abs(axis_length)

            # --- Draw 3 Cylinders along +Y, stepped in +X ---
            for i in range(num_cylinders):
                # Calculate the X-translation (0.0, 0.2, 0.4)
                x_pos = i * x_step

                glPushMatrix()
                
                # 1. Translate the cylinder to its position (X-step, Y=0, Z-offset)
                glTranslatef(x_pos, 0.0, z_offset)

                # 2. Rotate the cylinder from its default +Z axis to the +Y axis
                glRotatef(90, 1.0, 0.0, 0.0)
                
                # 3. Draw the cylinder using the pre-created 'quad' object
                gluCylinder(quad, radius, radius, length, 10, 2)
                
                glPopMatrix()

            # --- Cleanup ---
            # Delete the quadric object to free up resources
            gluDeleteQuadric(quad)

            # Reset color (optional, but good practice)
            glColor3f(1.0, 1.0, 1.0)



 
    def draw_background(self, image):

        # ⚠️ Do NOT flip here (we’ll flip via texture coordinates instead) 
        bg_image = Image.fromarray(image)
        ix, iy = bg_image.size
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)

        # --- Disable depth test for background ---
        glDisable(GL_DEPTH_TEST)

        # --- 2D projection for image ---
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.image_w, 0, self.image_h, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # --- Upload as texture ---
        texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)

        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        
        # 🧭 FIX: Use standard texture coordinates (0,0) at bottom-left vertex (0,0)
        # This removes the previous flip logic.
        glTexCoord2f(0.0, 0.0); glVertex2f(0, 0)         # Bottom-left (0, 0) gets texture bottom (Y=0)
        glTexCoord2f(1.0, 0.0); glVertex2f(self.image_w, 0)      # Bottom-right gets texture bottom (Y=0)
        glTexCoord2f(1.0, 1.0); glVertex2f(self.image_w, self.image_h)   # Top-right gets texture top (Y=1)
        glTexCoord2f(0.0, 1.0); glVertex2f(0, self.image_h)      # Top-left gets texture top (Y=1)
        
        glEnd()
        glDisable(GL_TEXTURE_2D)

        # --- Clean up ---
        glBindTexture(GL_TEXTURE_2D, 0)
        glDeleteTextures([texid])

        # --- Restore matrices ---
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        # --- Re-enable depth test for 3D ---
        glEnable(GL_DEPTH_TEST)


 
 
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
        # parameters.adaptiveThreshConstant = 7.0
        parameters.adaptiveThreshConstant = 10.0 # ลองเพิ่มค่า
        parameters.minMarkerPerimeterRate = 0.01 # ลดค่านี้เพื่อตรวจจับมาร์คเกอร์ที่เล็กมาก
        parameters.polygonalApproxAccuracyRate = 0.03 # ลดค่านี้เพื่อความแม่นยำในการตรวจจับขอบเขต

        height, width, channels = image.shape
        # print(height, width)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        projectMatrix = intrinsic2Project(self.cam_matrix, width, height, 0.01, 500.0)
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
                    
            # Store the 3D positions of the markers we care about.
            marker_positions = {}
            
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.models:
                    rvec = rvecs[i]
                    tvec = tvecs[i]
                    draw_axis(image, rvec, tvec, self.cam_matrix, zero_dist_coefs)# draw on undistorted image
                    # draw_axis(image, rvec, tvec, self.cam_matrix, self.dist_coefs)# draw on distorted image
                    
                    # --- Draw small red circle at (x=0.1, y=0, z=0) relative to the marker origin ---
                    point_3d = np.array([[0.15, 0.0, 0.0]], dtype=np.float32)  # 1 cm along X-axis
                    points_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, self.cam_matrix, zero_dist_coefs)
                    center_2d = tuple(points_2d[0][0].astype(int))
                    cv2.circle(image, center_2d, 5, (0, 0, 255), -1)  # Red circle in image
                    
                    if self.filter.update(tvec):
                        model_matrix = extrinsic2ModelView(rvec, tvec)
                        self.pre_extrinsicMatrix[marker_id] = model_matrix
                    else:
                        model_matrix = self.pre_extrinsicMatrix.get(marker_id)
                        
                    if model_matrix is not None:
                        glLoadMatrixf(model_matrix)  # sets the coordinate system for the next drawing call

                        # --- Draw the OpenGL axis ---
                        self.draw_axis_opengl(axis_length=0.1)

                        # --- Draw a small red sphere at (x=0.15, y=0, z=0) ---
                        glColor3f(1.0, 0.0, 0.0)  # Red color
                        glPushMatrix()
                        glTranslatef(0.15, 0.0, 0.0)  # Move 15 cm along X-axis of the marker
                        quad = gluNewQuadric()
                        gluSphere(quad, 0.005, 12, 12)  # Radius = 0.005 m (5 mm)
                        glPopMatrix()

                        # --- Draw the 3D model ---
                        glColor3f(1.0, 1.0, 1.0)  # Reset color
                        
                        # --- Draw the 3D model ---
                        # 1. DEFINE MATERIAL PROPERTIES FOR THE MODEL
                        # Example: Use a white diffuse color for the model
                        model_ambient_diffuse = [0.8, 0.8, 0.8, 1.0] # Light gray/near-white
                        model_specular = [0.2, 0.2, 0.2, 1.0]       # Small specular highlight
                        model_shininess = [10.0]                    # Low shininess

                        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, model_ambient_diffuse)
                        glMaterialfv(GL_FRONT, GL_SPECULAR, model_specular)
                        glMaterialfv(GL_FRONT, GL_SHININESS, model_shininess)

                        # 2. APPLY TRANSFORMATIONS
                        scale = self.model_scale_dict.get(marker_id, 0.01) # Default scale if not found
                        glPushMatrix()
                        glScaled(scale, scale, scale)
                        glTranslatef(self.translate_x, self.translate_y, self.translate_z)

                        # Ensure textures enabled if OBJ has textures
                        glEnable(GL_TEXTURE_2D)

                        # 3. DRAW MODEL
                        # If objloader uses glColor for materials, this will let them show.
                        glColor3f(1.0, 1.0, 1.0)
                        glCallList(self.models[marker_id].gl_list)

                        glDisable(GL_TEXTURE_2D)
                        glPopMatrix()
                        # Reset material to default (optional, but clean)
                        # glColor3f(1.0, 1.0, 1.0) # NOTE: glColor3f is ignored when GL_LIGHTING is on

                    # Store the 3D position of the detected marker.
                    marker_positions[marker_id] = tvecs[i][0]
                else:
                    marker_positions[marker_id] = np.zeros(3, dtype=np.float64)
                
            # --- distance calculation and drawing in ID0 frame ---
            # Check if both markers 0 and 3 are detected.
            if 0 in marker_positions and 3 in marker_positions:
                point0_3d_cam = marker_positions[0] # tvec for marker 0 (in Camera Frame)
                point3_3d_cam = marker_positions[3] # tvec for marker 3 (in Camera Frame)

                # Get rotation vector for marker 0
                try:
                    idx0 = np.where(ids.flatten() == 0)[0][0]
                    rvec0 = rvecs[idx0].reshape(3, 1)
                except Exception:
                    # Fallback if rvec0 is somehow missed (though it should be here)
                    rvec0 = np.zeros((3, 1))

                # 1. Calculate the rotation matrix R_cam_to_m0 (Rotation from Camera to Marker 0 frame)
                # R_m0_to_cam is the rotation from Marker 0 to Camera frame
                R_m0_to_cam, _ = cv2.Rodrigues(rvec0)
                R_cam_to_m0 = R_m0_to_cam.T # The inverse rotation

                # 2. Compute the position of Marker 4 RELATIVE to Marker 0, in Marker 0's frame
                # p_rel_cam = t4 - t0  (Vector from 0 to 4, in Camera Frame)
                p_rel_cam = point3_3d_cam - point0_3d_cam

                # p_rel_m0 = R_cam_to_m0 @ p_rel_cam (Vector from 0 to 4, in Marker 0 Frame)
                p_rel_m0 = R_cam_to_m0 @ p_rel_cam.reshape(3, 1)

                
                # 3. Extract the distances and calculate the total Euclidean distance
                dx = p_rel_m0[0, 0]
                dy = p_rel_m0[1, 0]
                dz = p_rel_m0[2, 0]
                
                # Total distance (Euclidean)
                distance_total = np.linalg.norm(p_rel_cam)

                # 4. Project the 3D points to 2D image coordinates to draw the line and text.
                # Use the midpoint of the camera vectors for the text label position.
                mid_point_3d_cam = (point0_3d_cam + point3_3d_cam) / 2
                
                r_vec_zero = np.zeros((3, 1))
                t_vec_zero = np.zeros((3, 1))
                
                # Project points to 2D
                points_2d, _ = cv2.projectPoints(
                    np.array([point0_3d_cam, point3_3d_cam, mid_point_3d_cam]), 
                    r_vec_zero, 
                    t_vec_zero, 
                    self.cam_matrix, 
                    zero_dist_coefs
                )
                
                # Extract the 2D points.
                point1_2d = tuple(points_2d[0][0].astype(int))
                point2_2d = tuple(points_2d[1][0].astype(int))
                mid_point_2d = tuple(points_2d[2][0].astype(int))

                # Draw a line between the two marker centers. BGR
                cv2.line(image, point1_2d, point2_2d, (128, 0, 128), 1)

                
                # --- NEW: draw projected purple line along marker0 X axis (length = dx) ---
                # point at (dx, 0, 0) in marker0 frame
                pt_dx_m0 = np.array([[dx], [0.0], [0.0]], dtype=np.float64)  # column vector
                # convert to camera frame: cam_pt = R_m0_to_cam * pt_m0 + t0
                cam_pt_dx = R_m0_to_cam @ pt_dx_m0 + point0_3d_cam.reshape(3,1)
                # project to 2D (camera-frame point, use zero rvec/tvec)
                proj_dx, _ = cv2.projectPoints(cam_pt_dx.reshape(1,3), r_vec_zero, t_vec_zero, self.cam_matrix, zero_dist_coefs)
                proj_dx2 = tuple(proj_dx[0][0].astype(int))

                # --- NEW: draw projected cyan line along marker0 Y axis (length = dy) ---
                pt_dy_m0 = np.array([[0.0], [dy], [0.0]], dtype=np.float64)
                cam_pt_dy = R_m0_to_cam @ pt_dy_m0 + point0_3d_cam.reshape(3,1)
                proj_dy, _ = cv2.projectPoints(cam_pt_dy.reshape(1,3), r_vec_zero, t_vec_zero, self.cam_matrix, zero_dist_coefs)
                proj_dy2 = tuple(proj_dy[0][0].astype(int))

                # origin in image is point0_2d (point1_2d variable above)
                origin2d = point1_2d

                # draw X-axis projection (purple)
                cv2.line(image, origin2d, proj_dx2, (255, 0, 255), 2)
                cv2.circle(image, proj_dx2, 4, (255,0,255), -1)

                # draw Y-axis projection (cyan)
                cv2.line(image, origin2d, proj_dy2, (255, 255, 0), 2)
                cv2.circle(image, proj_dy2, 4, (255,255,0), -1)

                # label dx and dy near each projected endpoint (offset to avoid overlap)
                mid_dx = ((origin2d[0] + proj_dx2[0]) // 2, (origin2d[1] + proj_dx2[1]) // 2 - 8)
                mid_dy = ((origin2d[0] + proj_dy2[0]) // 2 + 8, (origin2d[1] + proj_dy2[1]) // 2)
                cv2.putText(image, f"dx={dx:.2f}m", mid_dx, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 1, cv2.LINE_AA)
                cv2.putText(image, f"dy={dy:.2f}m", mid_dy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1, cv2.LINE_AA)
            
            
                
                # where you currently call draw_tube_opengl or draw simple GL cylinder:
                if 0 in marker_positions and 5 in marker_positions:
                    self.draw_tube_opengl(0,3,n_tube=3,radius=0.03,axis_length=dy) # marker id 0 to 3

            # ...end markers 0 and 3 ...
            
            # Check if both markers 0 and 4 are detected.
            if 0 in marker_positions and 4 in marker_positions:
                point0_3d_cam = marker_positions[0] # tvec for marker 0 (in Camera Frame)
                point4_3d_cam = marker_positions[4] # tvec for marker 4 (in Camera Frame)

                # Get rotation vector for marker 0
                try:
                    idx0 = np.where(ids.flatten() == 0)[0][0]
                    rvec0 = rvecs[idx0].reshape(3, 1)
                except Exception:
                    # Fallback if rvec0 is somehow missed (though it should be here)
                    rvec0 = np.zeros((3, 1))

                # 1. Calculate the rotation matrix R_cam_to_m0 (Rotation from Camera to Marker 0 frame)
                # R_m0_to_cam is the rotation from Marker 0 to Camera frame
                R_m0_to_cam, _ = cv2.Rodrigues(rvec0)
                R_cam_to_m0 = R_m0_to_cam.T # The inverse rotation

                # 2. Compute the position of Marker 4 RELATIVE to Marker 0, in Marker 0's frame
                # p_rel_cam = t4 - t0  (Vector from 0 to 4, in Camera Frame)
                p_rel_cam = point4_3d_cam - point0_3d_cam

                # p_rel_m0 = R_cam_to_m0 @ p_rel_cam (Vector from 0 to 4, in Marker 0 Frame)
                p_rel_m0 = R_cam_to_m0 @ p_rel_cam.reshape(3, 1)

                
                # 3. Extract the distances and calculate the total Euclidean distance
                dx = p_rel_m0[0, 0]
                dy = p_rel_m0[1, 0]
                dz = p_rel_m0[2, 0]
                
                # Total distance (Euclidean)
                distance_total = np.linalg.norm(p_rel_cam)

                # 4. Project the 3D points to 2D image coordinates to draw the line and text.
                # Use the midpoint of the camera vectors for the text label position.
                mid_point_3d_cam = (point0_3d_cam + point4_3d_cam) / 2
                
                r_vec_zero = np.zeros((3, 1))
                t_vec_zero = np.zeros((3, 1))
                
                # Project points to 2D
                points_2d, _ = cv2.projectPoints(
                    np.array([point0_3d_cam, point4_3d_cam, mid_point_3d_cam]), 
                    r_vec_zero, 
                    t_vec_zero, 
                    self.cam_matrix, 
                    zero_dist_coefs
                )
                
                # Extract the 2D points.
                point1_2d = tuple(points_2d[0][0].astype(int))
                point2_2d = tuple(points_2d[1][0].astype(int))
                mid_point_2d = tuple(points_2d[2][0].astype(int))

                # Draw a line between the two marker centers. BGR
                cv2.line(image, point1_2d, point2_2d, (128, 0, 128), 1)

                
                # --- NEW: draw projected purple line along marker0 X axis (length = dx) ---
                # point at (dx, 0, 0) in marker0 frame
                pt_dx_m0 = np.array([[dx], [0.0], [0.0]], dtype=np.float64)  # column vector
                # convert to camera frame: cam_pt = R_m0_to_cam * pt_m0 + t0
                cam_pt_dx = R_m0_to_cam @ pt_dx_m0 + point0_3d_cam.reshape(3,1)
                # project to 2D (camera-frame point, use zero rvec/tvec)
                proj_dx, _ = cv2.projectPoints(cam_pt_dx.reshape(1,3), r_vec_zero, t_vec_zero, self.cam_matrix, zero_dist_coefs)
                proj_dx2 = tuple(proj_dx[0][0].astype(int))

                # --- NEW: draw projected cyan line along marker0 Y axis (length = dy) ---
                pt_dy_m0 = np.array([[0.0], [dy], [0.0]], dtype=np.float64)
                cam_pt_dy = R_m0_to_cam @ pt_dy_m0 + point0_3d_cam.reshape(3,1)
                proj_dy, _ = cv2.projectPoints(cam_pt_dy.reshape(1,3), r_vec_zero, t_vec_zero, self.cam_matrix, zero_dist_coefs)
                proj_dy2 = tuple(proj_dy[0][0].astype(int))

                # origin in image is point0_2d (point1_2d variable above)
                origin2d = point1_2d

                # draw X-axis projection (purple)
                cv2.line(image, origin2d, proj_dx2, (255, 0, 255), 2)
                cv2.circle(image, proj_dx2, 4, (255,0,255), -1)

                # draw Y-axis projection (cyan)
                cv2.line(image, origin2d, proj_dy2, (255, 255, 0), 2)
                cv2.circle(image, proj_dy2, 4, (255,255,0), -1)

                # label dx and dy near each projected endpoint (offset to avoid overlap)
                mid_dx = ((origin2d[0] + proj_dx2[0]) // 2, (origin2d[1] + proj_dx2[1]) // 2 - 32)
                mid_dy = ((origin2d[0] + proj_dy2[0]) // 2 + 8, (origin2d[1] + proj_dy2[1]) // 2)
                cv2.putText(image, f"dx={dx:.2f}m", mid_dx, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 1, cv2.LINE_AA)
                cv2.putText(image, f"dy={dy:.2f}m", mid_dy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1, cv2.LINE_AA)
            # ...end markers 0 and 4 ...

            # Check if both markers 0 and 5 are detected.
            if 0 in marker_positions and 5 in marker_positions:
                point0_3d_cam = marker_positions[0] # tvec for marker 0 (in Camera Frame)
                point5_3d_cam = marker_positions[5] # tvec for marker 4 (in Camera Frame)

                # Get rotation vector for marker 0
                try:
                    idx0 = np.where(ids.flatten() == 0)[0][0]
                    rvec0 = rvecs[idx0].reshape(3, 1)
                except Exception:
                    # Fallback if rvec0 is somehow missed (though it should be here)
                    rvec0 = np.zeros((3, 1))

                # 1. Calculate the rotation matrix R_cam_to_m0 (Rotation from Camera to Marker 0 frame)
                # R_m0_to_cam is the rotation from Marker 0 to Camera frame
                R_m0_to_cam, _ = cv2.Rodrigues(rvec0)
                R_cam_to_m0 = R_m0_to_cam.T # The inverse rotation

                # 2. Compute the position of Marker 4 RELATIVE to Marker 0, in Marker 0's frame
                # p_rel_cam = t4 - t0  (Vector from 0 to 4, in Camera Frame)
                p_rel_cam = point5_3d_cam - point0_3d_cam

                # p_rel_m0 = R_cam_to_m0 @ p_rel_cam (Vector from 0 to 4, in Marker 0 Frame)
                p_rel_m0 = R_cam_to_m0 @ p_rel_cam.reshape(3, 1)

                
                # 3. Extract the distances and calculate the total Euclidean distance
                dx = p_rel_m0[0, 0]
                dy = p_rel_m0[1, 0]
                dz = p_rel_m0[2, 0]
                
                # Total distance (Euclidean)
                distance_total = np.linalg.norm(p_rel_cam)

                # 4. Project the 3D points to 2D image coordinates to draw the line and text.
                # Use the midpoint of the camera vectors for the text label position.
                mid_point_3d_cam = (point0_3d_cam + point5_3d_cam) / 2
                
                r_vec_zero = np.zeros((3, 1))
                t_vec_zero = np.zeros((3, 1))
                
                # Project points to 2D
                points_2d, _ = cv2.projectPoints(
                    np.array([point0_3d_cam, point5_3d_cam, mid_point_3d_cam]), 
                    r_vec_zero, 
                    t_vec_zero, 
                    self.cam_matrix, 
                    zero_dist_coefs
                )
                
                # Extract the 2D points.
                point1_2d = tuple(points_2d[0][0].astype(int))
                point2_2d = tuple(points_2d[1][0].astype(int))
                mid_point_2d = tuple(points_2d[2][0].astype(int))

                # Draw a line between the two marker centers. BGR
                cv2.line(image, point1_2d, point2_2d, (128, 0, 128), 1)

                
                # --- NEW: draw projected purple line along marker0 X axis (length = dx) ---
                # point at (dx, 0, 0) in marker0 frame
                pt_dx_m0 = np.array([[dx], [0.0], [0.0]], dtype=np.float64)  # column vector
                # convert to camera frame: cam_pt = R_m0_to_cam * pt_m0 + t0
                cam_pt_dx = R_m0_to_cam @ pt_dx_m0 + point0_3d_cam.reshape(3,1)
                # project to 2D (camera-frame point, use zero rvec/tvec)
                proj_dx, _ = cv2.projectPoints(cam_pt_dx.reshape(1,3), r_vec_zero, t_vec_zero, self.cam_matrix, zero_dist_coefs)
                proj_dx2 = tuple(proj_dx[0][0].astype(int))

                # --- NEW: draw projected cyan line along marker0 Y axis (length = dy) ---
                pt_dy_m0 = np.array([[0.0], [dy], [0.0]], dtype=np.float64)
                cam_pt_dy = R_m0_to_cam @ pt_dy_m0 + point0_3d_cam.reshape(3,1)
                proj_dy, _ = cv2.projectPoints(cam_pt_dy.reshape(1,3), r_vec_zero, t_vec_zero, self.cam_matrix, zero_dist_coefs)
                proj_dy2 = tuple(proj_dy[0][0].astype(int))

                # origin in image is point0_2d (point1_2d variable above)
                origin2d = point1_2d

                # draw X-axis projection (purple)
                cv2.line(image, origin2d, proj_dx2, (255, 0, 255), 2)
                cv2.circle(image, proj_dx2, 4, (255,0,255), -1)

                # draw Y-axis projection (cyan)
                cv2.line(image, origin2d, proj_dy2, (255, 255, 0), 2)
                cv2.circle(image, proj_dy2, 4, (255,255,0), -1)

                # label dx and dy near each projected endpoint (offset to avoid overlap)
                mid_dx = ((origin2d[0] + proj_dx2[0]) // 2, (origin2d[1] + proj_dx2[1]) // 2 - 8)
                mid_dy = ((origin2d[0] + proj_dy2[0]) // 2 + 8, (origin2d[1] + proj_dy2[1]) // 2)
                cv2.putText(image, f"dx={dx:.2f}m", mid_dx, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 1, cv2.LINE_AA)
                cv2.putText(image, f"dy={dy:.2f}m", mid_dy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1, cv2.LINE_AA)
            
              
            # ...end markers 0 and 5 ...
            
            # Check if both markers 4 and 3 are detected.
            if 4 in marker_positions and 3 in marker_positions:
                point4_3d_cam = marker_positions[4] # tvec for marker 0 (in Camera Frame)
                point3_3d_cam = marker_positions[3] # tvec for marker 3 (in Camera Frame)

                # Get rotation vector for marker 0
                try:
                    idx0 = np.where(ids.flatten() == 0)[0][0]
                    rvec0 = rvecs[idx0].reshape(3, 1)
                except Exception:
                    # Fallback if rvec0 is somehow missed (though it should be here)
                    rvec0 = np.zeros((3, 1))

                # 1. Calculate the rotation matrix R_cam_to_m0 (Rotation from Camera to Marker 0 frame)
                # R_m0_to_cam is the rotation from Marker 0 to Camera frame
                R_m0_to_cam, _ = cv2.Rodrigues(rvec0)
                R_cam_to_m0 = R_m0_to_cam.T # The inverse rotation

                # 2. Compute the position of Marker 4 RELATIVE to Marker 0, in Marker 0's frame
                # p_rel_cam = t4 - t0  (Vector from 0 to 4, in Camera Frame)
                p_rel_cam = point3_3d_cam - point4_3d_cam

                # p_rel_m0 = R_cam_to_m0 @ p_rel_cam (Vector from 0 to 4, in Marker 0 Frame)
                p_rel_m0 = R_cam_to_m0 @ p_rel_cam.reshape(3, 1)

                
                # 3. Extract the distances and calculate the total Euclidean distance
                dx = p_rel_m0[0, 0]
                dy = p_rel_m0[1, 0]
                dz = p_rel_m0[2, 0]
                
                # Total distance (Euclidean)
                distance_total = np.linalg.norm(p_rel_cam)

                # 4. Project the 3D points to 2D image coordinates to draw the line and text.
                # Use the midpoint of the camera vectors for the text label position.
                mid_point_3d_cam = (point4_3d_cam + point3_3d_cam) / 2
                
                r_vec_zero = np.zeros((3, 1))
                t_vec_zero = np.zeros((3, 1))
                
                # Project points to 2D
                points_2d, _ = cv2.projectPoints(
                    np.array([point4_3d_cam, point3_3d_cam, mid_point_3d_cam]), 
                    r_vec_zero, 
                    t_vec_zero, 
                    self.cam_matrix, 
                    zero_dist_coefs
                )
                
                # Extract the 2D points.
                point1_2d = tuple(points_2d[0][0].astype(int))
                point2_2d = tuple(points_2d[1][0].astype(int))
                mid_point_2d = tuple(points_2d[2][0].astype(int))

                # Draw a line between the two marker centers. BGR
                cv2.line(image, point1_2d, point2_2d, (128, 0, 128), 1)

                
                # --- NEW: draw projected purple line along marker0 X axis (length = dx) ---
                # point at (dx, 0, 0) in marker0 frame
                pt_dx_m0 = np.array([[dx], [0.0], [0.0]], dtype=np.float64)  # column vector
                # convert to camera frame: cam_pt = R_m0_to_cam * pt_m0 + t0
                cam_pt_dx = R_m0_to_cam @ pt_dx_m0 + point4_3d_cam.reshape(3,1)
                # project to 2D (camera-frame point, use zero rvec/tvec)
                proj_dx, _ = cv2.projectPoints(cam_pt_dx.reshape(1,3), r_vec_zero, t_vec_zero, self.cam_matrix, zero_dist_coefs)
                proj_dx2 = tuple(proj_dx[0][0].astype(int))

                # --- NEW: draw projected cyan line along marker0 Y axis (length = dy) ---
                pt_dy_m0 = np.array([[0.0], [dy], [0.0]], dtype=np.float64)
                cam_pt_dy = R_m0_to_cam @ pt_dy_m0 + point4_3d_cam.reshape(3,1)
                proj_dy, _ = cv2.projectPoints(cam_pt_dy.reshape(1,3), r_vec_zero, t_vec_zero, self.cam_matrix, zero_dist_coefs)
                proj_dy2 = tuple(proj_dy[0][0].astype(int))

                # origin in image is point0_2d (point1_2d variable above)
                origin2d = point1_2d

                # draw X-axis projection (purple)
                cv2.line(image, origin2d, proj_dx2, (255, 0, 255), 2)
                cv2.circle(image, proj_dx2, 4, (255,0,255), -1)

                # draw Y-axis projection (cyan)
                cv2.line(image, origin2d, proj_dy2, (255, 255, 0), 2)
                cv2.circle(image, proj_dy2, 4, (255,255,0), -1)

                # label dx and dy near each projected endpoint (offset to avoid overlap)
                mid_dx = ((origin2d[0] + proj_dx2[0]) // 2, (origin2d[1] + proj_dx2[1]) // 2 - 8)
                mid_dy = ((origin2d[0] + proj_dy2[0]) // 2 + 8, (origin2d[1] + proj_dy2[1]) // 2)
                cv2.putText(image, f"dx={dx:.2f}m", mid_dx, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 1, cv2.LINE_AA)
                cv2.putText(image, f"dy={dy:.2f}m", mid_dy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1, cv2.LINE_AA)
            
            
                
                # where you currently call draw_tube_opengl or draw simple GL cylinder:
                if 0 in marker_positions and 5 in marker_positions:
                    self.draw_tube_opengl(4,3,n_tube=2,radius=0.04,axis_length=dy) # marker id 4 to 3

            # ...end markers 0 and 3 ...

                
            # --- Top-view frame visualization (marker0 reference, shows height too) ---
            if 0 in marker_positions:
                # 1️⃣  Get marker0 pose
                t0 = marker_positions[0].reshape(3)
                try:
                    idx0 = np.where(ids.flatten() == 0)[0][0]
                    rvec0 = rvecs[idx0].reshape(3)
                except Exception:
                    rvec0 = np.zeros(3)
                R_m0_to_cam, _ = cv2.Rodrigues(rvec0)
                R_cam_to_m0 = R_m0_to_cam.T  # camera → marker0 rotation

                # 2️⃣  Make blank canvas
                map_size = 600
                scale = 300  # pixels per meter
                center = (map_size // 2, map_size // 2)
                map_img = np.ones((map_size, map_size, 3), np.uint8) * 255

                # grid
                for g in range(0, map_size, 50):
                    cv2.line(map_img, (g, 0), (g, map_size), (230,230,230), 1)
                    cv2.line(map_img, (0, g), (map_size, g), (230,230,230), 1)

                # origin (ID0)
                cv2.circle(map_img, center, 6, (0,0,255), -1)
                cv2.putText(map_img, "ID0", (center[0]+8, center[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,50), 1)

                # 3️⃣  For each marker, compute its coordinates in marker0 frame
                for marker_id, pos_cam in marker_positions.items():
                    t_i = pos_cam.reshape(3)
                    p_rel = R_cam_to_m0.dot(t_i - t0)  # in marker0 frame

                    x_m0, y_m0, z_m0 = p_rel  # local coordinates (m)

                    # Convert to pixel coords for top view (XY plane)
                    px = int(center[0] + x_m0 * scale)
                    py = int(center[1] - y_m0 * scale)

                    # Encode height (Z) by color: blue=below, red=above
                    if abs(z_m0) < 0.01:
                        color = (0, 255, 0)   # near plane → green
                    elif z_m0 > 0:
                        color = (0, 0, 255)   # above → red
                    else:
                        color = (255, 0, 0)   # below → blue

                    cv2.circle(map_img, (px, py), 6, color, -1)
                    cv2.putText(map_img,
                                f"ID{marker_id} z={z_m0:.3f}m",
                                (px + 8, py - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (30, 30, 30), 1)

                    # Optional: draw a small X-axis arrow in marker0 frame
                    try:
                        idx = np.where(ids.flatten() == marker_id)[0][0]
                        rvec_i = rvecs[idx].reshape(3)
                        R_mi_to_cam, _ = cv2.Rodrigues(rvec_i)
                        x_axis_cam = R_mi_to_cam @ np.array([1.0, 0.0, 0.0])
                        x_axis_m0 = R_cam_to_m0 @ x_axis_cam
                        end = p_rel + 0.05 * x_axis_m0
                        px2 = int(center[0] + end[0] * scale)
                        py2 = int(center[1] - end[1] * scale)
                        cv2.line(map_img, (px, py), (px2, py2), (100, 100, 255), 2)
                    except Exception:
                        pass

                # 4️⃣  Display the top view
                cv2.imshow("Top View (marker0 XY, Z color)", map_img)

                #***********************************************************************************
                
                #***********************************************************************************
                
                
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
        # with np.load('calibration_results.npz') as file:
        with np.load('calibration_results_frontcam.npz') as file:
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
        1: './Models/Barn/ban.obj',
        0: './Models/INV/luna2000_213ktl.obj',
        2: './Models/Monster/Sinbad_4_000001.obj',
        3: './Models/Wireway/wireway1.obj',
        4: './Models/EBox/EBox.obj',
        5: './Models/Button/button.obj',
        5: './Models/Sphere/sphere.obj'
    }
    model_scale_dict = {
        1: 0.02,  # scale for marker 0
        0: 1,
        2: 0.1,
        3: 1,
        4: 1,
        5: 0.1,
        6: 0.1
    }
    ar_instance = AR_render(cam_matrix, dist_coeff, id_to_model, model_scale_dict)
    
    fy = cam_matrix[1, 1]
    image_height = ar_instance.image_h
    image_width = ar_instance.image_w
    fovy = 2 * np.arctan(image_height / (2 * fy)) * 180 / np.pi
    aspect = image_width / image_height
    print(fovy, aspect) # use for gluPerspective
    
    ar_instance.run()