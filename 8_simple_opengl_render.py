import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from stl import mesh

# --- 1. Load 3D Model from STL file ---
try:
    # Load the STL file. Make sure your STL file is in the same directory as this script.
    model_mesh = mesh.Mesh.from_file('model2.stl')
    
    # Extract vertices from the STL mesh
    model_vertices = model_mesh.vectors.reshape(-1, 3)
    
    # Normalize the model's coordinates for a good starting size and position.
    max_coords = np.max(model_vertices, axis=0)
    min_coords = np.min(model_vertices, axis=0)
    model_size = max_coords - min_coords
    scale = 1.0 / max(model_size)
    
    # Center the model and apply scaling.
    model_vertices = (model_vertices - min_coords - model_size / 2) * scale
    
    # Create the transformed faces array for drawing.
    model_faces = model_vertices.reshape(-1, 3, 3)
    
    print("3D model loaded and prepared.")

except FileNotFoundError:
    print("Error: 'model.stl' not found.")
    print("Please place an STL file named 'model.stl' in the same directory as this script.")
    exit()

# --- 2. OpenGL Setup ---
def init_gl(width, height):
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    
    # Set up the light source properties
    glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, 5, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
    
    # Set the material properties
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

# --- 3. Main Loop for Rendering ---
def main():
    pygame.init()
    display_width = 800
    display_height = 600
    pygame.display.set_mode((display_width, display_height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Pure OpenGL STL Renderer")
    init_gl(display_width, display_height)

    # Move the camera back to view the model
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
    
    rotation_angle = 0
    
    print("Press 'q' to quit.")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_q):
                pygame.quit()
                return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Position and rotate the model
        glTranslate(0, 0, -5) # Move the model back
        glRotate(rotation_angle, 1, 1, 0) # Rotate the model
        rotation_angle += 1 # Increment the angle for continuous rotation

        # Draw the model
        glColor3f(0.0, 0.5, 0.9) # A nice blue color
        glBegin(GL_TRIANGLES)
        for face in model_faces:
            glVertex3f(face[0][0], face[0][1], face[0][2])
            glVertex3f(face[1][0], face[1][1], face[1][2])
            glVertex3f(face[2][0], face[2][1], face[2][2])
        glEnd()

        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == '__main__':
    main()
