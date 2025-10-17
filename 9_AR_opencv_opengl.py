from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
import cv2.aruco as aruco
from PIL import Image
import numpy as np
import sys

from tools.Visualize import draw_axis
from tools.objloader import OBJ
from tools.matrixTrans import extrinsic2ModelView, intrinsic2Project
from tools.Filter import Filter


class ARRender:
    """
    Cleaner organization of the original AR_render class.
    Keeps existing behavior: background texture, marker detection,
    model rendering, ID0 filtering behavior, top-view and distance annotations.
    """

    def __init__(self, camera_matrix, dist_coefs, id_to_model, model_scale_dict):
        # Camera / OpenGL state
        self.cam_matrix = camera_matrix
        self.dist_coefs = dist_coefs
        self.id_to_model = id_to_model
        self.model_scale_dict = model_scale_dict

        # Video source (keep original IP stream default)
        src = "http://192.168.1.59:1984/videostream.cgi?user=admin&pwd=88888888"
        self.webcam = cv2.VideoCapture(src)
        self.image_w, self.image_h = map(int, (self.webcam.get(3), self.webcam.get(4)))
        print("Video size:", self.image_w, "x", self.image_h)

        # Init GL FIRST so any GL calls (glGenLists in OBJ) have a valid context
        self.init_opengl(self.image_w, self.image_h)

        # Now load models (OBJ may call glGenLists / other GL functions)
        self.models = {mid: OBJ(path, swapyz=True) for mid, path in id_to_model.items()}

        # Models and transforms
        self.pre_extrinsicMatrix = {}
        self.translate_x = self.translate_y = self.translate_z = 0.0

        # Filter (keeps existing behavior of Filter.update used in original code)
        self.filter = Filter()

    # -----------------------
    # OpenGL / Window setup
    # -----------------------
    def init_opengl(self, width, height, pos_x=500, pos_y=500, window_name=b"Aruco Demo"):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(pos_x, pos_y)
        self.window_id = glutCreateWindow(window_name)
        glutDisplayFunc(self.draw_scene)
        glutIdleFunc(self.draw_scene)
        glutKeyboardFunc(self.keyboard_listener)

        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glShadeModel(GL_SMOOTH)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)

        # lighting placeholder (kept from original)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1))

    # -----------------------
    # Main loop entry point
    # -----------------------
    def run(self):
        glutMainLoop()

    # -----------------------
    # Frame read + scene
    # -----------------------
    def draw_scene(self):
        ok, image_raw = self.webcam.read()
        if not ok:
            return

        # Undistort for rendering / projection alignment
        image = cv2.undistort(image_raw, self.cam_matrix, self.dist_coefs, None, self.cam_matrix)

        # Clear buffers and render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Background (draw first, no depth)
        self.draw_background(image)

        # Setup projection from intrinsics (near/far adjusted to allow >1m)
        h, w = image.shape[:2]
        proj = intrinsic2Project(self.cam_matrix, w, h, 0.01, 500.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixf(proj)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Draw AR objects and 2D annotations
        self.draw_objects(image)

        glutSwapBuffers()

    # -----------------------
    # Background as textured quad
    # -----------------------
    def draw_background(self, image):
        # Convert to PIL and ensure correct byte order for glTexImage2D
        # Keep orientation consistent with texture coords used below
        pil = Image.fromarray(image)
        ix, iy = pil.size
        bg_bytes = pil.tobytes("raw", "BGRX", 0, -1)

        # Disable depth so quad is always behind
        glDisable(GL_DEPTH_TEST)

        # 2D orthographic projection for the textured quad
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.image_w, 0, self.image_h, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Upload texture and draw quad
        texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_bytes)

        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        # texture coordinates: (0,0) bottom-left -> matches vertex (0,0)
        glTexCoord2f(0.0, 0.0); glVertex2f(0, 0)
        glTexCoord2f(1.0, 0.0); glVertex2f(self.image_w, 0)
        glTexCoord2f(1.0, 1.0); glVertex2f(self.image_w, self.image_h)
        glTexCoord2f(0.0, 1.0); glVertex2f(0, self.image_h)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        # cleanup
        glBindTexture(GL_TEXTURE_2D, 0)
        glDeleteTextures([texid])

        # restore matrices and enable depth
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)

    # -----------------------
    # High level marker processing & drawing
    # -----------------------
    def draw_objects(self, image, mark_size=0.08):
        # detect markers
        ar_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        params = aruco.DetectorParameters()
        params.adaptiveThreshConstant = 10.0
        params.minMarkerPerimeterRate = 0.01
        params.polygonalApproxAccuracyRate = 0.03

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, ar_dict, parameters=params)

        # projection matrix already set by draw_scene

        if ids is None or corners is None:
            cv2.imshow("Frame", image)
            cv2.waitKey(20)
            return

        # Estimate poses using zero distortion for accuracy on undistorted image
        zero_dist = np.zeros(5)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, mark_size, self.cam_matrix, zero_dist)

        # Special handling for marker ID 0 (keeps original filter behavior)
        if 0 in ids:
            idx0 = int(np.where(ids.flatten() == 0)[0][0])
            rvec0 = rvecs[idx0]
            tvec0 = tvecs[idx0]
            if self.filter.update(tvec0):
                model0 = extrinsic2ModelView(rvec0, tvec0)
                self.pre_extrinsicMatrix[0] = model0
            else:
                model0 = self.pre_extrinsicMatrix.get(0)
            if model0 is not None:
                glLoadMatrixf(model0)
                # mark world origin with a purple point
                glPointSize(10.0)
                glColor3f(1.0, 0.0, 1.0)
                glBegin(GL_POINTS)
                glVertex3f(0.0, 0.0, 0.0)
                glEnd()
                glColor3f(1.0, 1.0, 1.0)

        # Iterate markers: draw axes, small point, model
        marker_positions = {}
        for i, mid in enumerate(ids.flatten()):
            if mid not in self.models:
                continue

            rvec = rvecs[i]
            tvec = tvecs[i]

            # draw OpenCV axis overlay (on undistorted image)
            draw_axis(image, rvec, tvec, self.cam_matrix, zero_dist)

            # draw small red point projected from a marker-local 3D point
            p_local = np.array([[0.15, 0.0, 0.0]], dtype=np.float32)
            pts2d, _ = cv2.projectPoints(p_local, rvec, tvec, self.cam_matrix, zero_dist)
            center_2d = tuple(pts2d[0][0].astype(int))
            cv2.circle(image, center_2d, 5, (0, 0, 255), -1)

            # apply filter behavior per original code
            if self.filter.update(tvec):
                model_matrix = extrinsic2ModelView(rvec, tvec)
                self.pre_extrinsicMatrix[mid] = model_matrix
            else:
                model_matrix = self.pre_extrinsicMatrix.get(mid)

            if model_matrix is not None:
                glLoadMatrixf(model_matrix)
                # draw OpenGL axis and a small sphere marker
                self.draw_axis_opengl(axis_length=0.1)
                glColor3f(1.0, 0.0, 0.0)
                glPushMatrix()
                glTranslatef(0.15, 0.0, 0.0)
                quad = gluNewQuadric()
                gluSphere(quad, 0.005, 12, 12)
                glPopMatrix()
                glColor3f(1.0, 1.0, 1.0)

                # draw model with per-id scale; use push/pop to avoid matrix accumulation
                glPushMatrix()
                scale = self.model_scale_dict.get(mid, 0.01)
                glScaled(scale, scale, scale)
                glTranslatef(self.translate_x, self.translate_y, self.translate_z)
                glCallList(self.models[mid].gl_list)
                glPopMatrix()

            # store camera-frame translation for later top-view/distance calculations
            marker_positions[mid] = tvecs[i][0]

        # If both ID0 and ID1 are present, compute relative vector and draw annotated lines
        if 0 in marker_positions and 1 in marker_positions:
            self._draw_marker_distance_annotations(image, ids, rvecs, tvecs, marker_positions)

        # draw top view (marker0 reference) if marker0 present
        if 0 in marker_positions:
            self._draw_top_view(ids, rvecs, marker_positions)

        # final 2D display
        cv2.imshow("Frame", image)
        cv2.waitKey(20)

    # -----------------------
    # Distance / projection helpers
    # -----------------------
    def _draw_marker_distance_annotations(self, image, ids, rvecs, tvecs, marker_positions):
        # Prepare rotation for marker0
        idx0 = int(np.where(ids.flatten() == 0)[0][0])
        try:
            rvec0 = rvecs[idx0].reshape(3, 1)
        except Exception:
            rvec0 = np.zeros((3, 1))
        R_m0_to_cam, _ = cv2.Rodrigues(rvec0)
        R_cam_to_m0 = R_m0_to_cam.T

        p0 = marker_positions[0].reshape(3, 1)
        p1 = marker_positions[1].reshape(3, 1)
        p_rel_cam = p1 - p0
        p_rel_m0 = R_cam_to_m0 @ p_rel_cam

        dx = float(p_rel_m0[0, 0])
        dy = float(p_rel_m0[1, 0])
        dz = float(p_rel_m0[2, 0])
        total_dist = np.linalg.norm(p_rel_cam)

        # project endpoints and midpoint (use zero rvec/tvec because points already in camera frame)
        mid3 = ((p0 + p1) / 2).reshape(3,)
        pts2d, _ = cv2.projectPoints(np.array([p0.reshape(3,), p1.reshape(3,), mid3]),
                                     np.zeros((3, 1)), np.zeros((3, 1)), self.cam_matrix, np.zeros(5))
        pt0_2d = tuple(pts2d[0][0].astype(int))
        pt1_2d = tuple(pts2d[1][0].astype(int))
        mid_2d = tuple(pts2d[2][0].astype(int))

        # line between centers
        cv2.line(image, pt0_2d, pt1_2d, (128, 0, 128), 2)
        cv2.putText(image, f"Total: {total_dist:.2f}m", (mid_2d[0], mid_2d[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2, cv2.LINE_AA)

        # project marker0-local (dx,0,0) and (0,dy,0) into image
        # compute their camera-frame coordinates: cam_pt = R_m0_to_cam @ pt_m0 + p0
        pt_dx_m0 = np.array([[dx], [0.0], [0.0]])
        cam_dx = (R_m0_to_cam @ pt_dx_m0) + p0
        proj_dx, _ = cv2.projectPoints(cam_dx.reshape(1, 3), np.zeros((3, 1)), np.zeros((3, 1)), self.cam_matrix, np.zeros(5))
        proj_dx2 = tuple(proj_dx[0][0].astype(int))

        pt_dy_m0 = np.array([[0.0], [dy], [0.0]])
        cam_dy = (R_m0_to_cam @ pt_dy_m0) + p0
        proj_dy, _ = cv2.projectPoints(cam_dy.reshape(1, 3), np.zeros((3, 1)), np.zeros((3, 1)), self.cam_matrix, np.zeros(5))
        proj_dy2 = tuple(proj_dy[0][0].astype(int))

        origin2d = pt0_2d

        # draw X-axis projection (purple) and Y-axis projection (cyan)
        cv2.line(image, origin2d, proj_dx2, (255, 0, 255), 2)
        cv2.circle(image, proj_dx2, 4, (255, 0, 255), -1)
        cv2.putText(image, f"dx={dx:.2f}m", ((origin2d[0]+proj_dx2[0])//2, (origin2d[1]+proj_dx2[1])//2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

        cv2.line(image, origin2d, proj_dy2, (255, 255, 0), 2)
        cv2.circle(image, proj_dy2, 4, (255, 255, 0), -1)
        cv2.putText(image, f"dy={dy:.2f}m", ((origin2d[0]+proj_dy2[0])//2 + 8, (origin2d[1]+proj_dy2[1])//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    # -----------------------
    # Top-view visualization (marker0 as origin)
    # -----------------------
    def _draw_top_view(self, ids, rvecs, marker_positions):
        # prepare marker0 pose
        t0 = marker_positions[0].reshape(3)
        try:
            idx0 = int(np.where(ids.flatten() == 0)[0][0])
            rvec0 = rvecs[idx0].reshape(3)
        except Exception:
            rvec0 = np.zeros(3)
        R_m0_to_cam, _ = cv2.Rodrigues(rvec0)
        R_cam_to_m0 = R_m0_to_cam.T

        # top-view canvas
        map_size = 600
        scale = 300  # pixels per meter
        center = (map_size // 2, map_size // 2)
        map_img = np.ones((map_size, map_size, 3), np.uint8) * 255

        # grid and origin
        for g in range(0, map_size, 50):
            cv2.line(map_img, (g, 0), (g, map_size), (230, 230, 230), 1)
            cv2.line(map_img, (0, g), (map_size, g), (230, 230, 230), 1)
        cv2.circle(map_img, center, 6, (0, 0, 255), -1)
        cv2.putText(map_img, "ID0", (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

        for mid, pos_cam in marker_positions.items():
            t_i = pos_cam.reshape(3)
            p_rel = R_cam_to_m0.dot(t_i - t0)
            x_m0, y_m0, z_m0 = p_rel
            px = int(center[0] + x_m0 * scale)
            py = int(center[1] - y_m0 * scale)
            if abs(z_m0) < 0.01:
                color = (0, 255, 0)
            elif z_m0 > 0:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            cv2.circle(map_img, (px, py), 6, color, -1)
            cv2.putText(map_img, f"ID{mid} z={z_m0:.3f}m", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)

            # small X-axis arrow (marker local)
            try:
                idx = int(np.where(ids.flatten() == mid)[0][0])
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

        cv2.imshow("Top View (marker0 XY, Z color)", map_img)

    # -----------------------
    # Keyboard
    # -----------------------
    def keyboard_listener(self, key, x, y):
        try:
            k = key.decode("utf-8")
        except Exception:
            return
        if k == "=":
            # keep compatibility if there is a per-instance model_scale variable
            pass
        elif k == "-":
            pass
        elif k == "w":
            self.translate_x -= 0.1
        elif k == "s":
            self.translate_x += 0.1
        elif k == "a":
            self.translate_y -= 0.1
        elif k == "d":
            self.translate_y += 0.1


if __name__ == "__main__":
    # load calibration if available, otherwise fallback to defaults (kept from original)
    try:
        with np.load("calibration_results.npz") as f:
            cam_matrix = f["camera_matrix"]
            dist_coeff = f["dist_coeffs"]
            print("Calibration loaded.")
    except FileNotFoundError:
        print("Using fallback calibration.")
        cam_matrix = np.array([
            [963.4519793109993, 0, 647.0863663141905],
            [0, 966.0565298361108, 352.23753011981177],
            [0, 0, 1],
        ])
        dist_coeff = np.array([-0.15259701966137876, 0.6092617145206677, 0.0007901395004658092, 0.0026990411152102638, -0.6577414700462231])

    id_to_model = {
        0: "./Models/Barn/ban.obj",
        1: "./Models/INV/INV.obj",
        2: "./Models/Monster/Sinbad_4_000001.obj",
        3: "./Models/Button/model.obj",
        4: "./Models/EBox/EBox.obj",
    }
    model_scale_dict = {0: 0.01, 1: 1, 2: 1, 3: 0.1, 4: 1}

    ar = ARRender(cam_matrix, dist_coeff, id_to_model, model_scale_dict)

    # print estimated fovy/aspect for debugging (same method as original)
    fy = cam_matrix[1, 1]
    image_height = ar.image_h
    image_width = ar.image_w
    fovy = 2 * np.arctan(image_height / (2 * fy)) * 180 / np.pi
    aspect = image_width / image_height
    print("fovy, aspect:", fovy, aspect)

    ar.run()