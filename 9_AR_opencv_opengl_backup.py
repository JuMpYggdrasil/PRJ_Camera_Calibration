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
        self.webcam = cv2.VideoCapture(0)
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
        _, image = self.webcam.read()# get image from webcam camera.
        self.draw_background(image)  # draw background
        # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.draw_objects(image, mark_size = 0.06) # draw the 3D objects.
        glutSwapBuffers()
    
        
        # TODO add close button
        # key = cv2.waitKey(20)
        
       
        
 
 
 
    def draw_background(self, image):
        """[Draw the background and tranform to opengl format]
        
        Arguments:
            image {[np.array]} -- [frame from your camera]
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Setting background image project_matrix and model_matrix.
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(33.7, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
     
        # Convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = Image.fromarray(bg_image)     
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)
  
  
        # Create background texture
        texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)
                
        glTranslatef(0.0,0.0,-10.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 4.0,  3.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-4.0,  3.0, 0.0)
        glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)
 
 
 
    def draw_objects(self, image, mark_size=0.01):
        """[draw models with opengl]
        
        Arguments:
            image {[np.array]} -- [frame from your camera]
        
        Keyword Arguments:
            mark_size {float} -- [aruco mark size: unit is meter] (default: {0.01})
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
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, mark_size, self.cam_matrix, self.dist_coefs)
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.models:
                    rvec = rvecs[i]
                    tvec = tvecs[i]
                    draw_axis(image, rvec, tvec, self.cam_matrix, self.dist_coefs)
                    if self.filter.update(tvec):
                        model_matrix = extrinsic2ModelView(rvec, tvec)
                        self.pre_extrinsicMatrix[marker_id] = model_matrix
                    else:
                        model_matrix = self.pre_extrinsicMatrix.get(marker_id)
                    if model_matrix is not None:
                        glLoadMatrixf(model_matrix)
                        scale = self.model_scale_dict.get(marker_id, 0.01)  # Default scale if not found
                        glScaled(scale, scale, scale)
                        glTranslatef(self.translate_x, self.translate_y, self.translate_z)
                        glCallList(self.models[marker_id].gl_list)
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