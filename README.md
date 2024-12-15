# ZCude
import cv2
import numpy as np
import mediapipe as mp
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Global variables for controlling zoom and rotation
zoom_factor = 1.0  # Initial zoom factor
rotation_x, rotation_y, rotation_z = 0.0, 0.0, 0.0  # Rotation angles for X, Y, and Z axis
previous_wrist_position = None
previous_index_position = None

# Timing variables to check hand movement
last_movement_time = time.time()  # Track last time of movement
movement_timeout = 2  # Timeout in seconds after which we consider both hands stopped
hand_movement_threshold = 0.01  # Threshold for movement detection (reduce noise)

# Function to render a 3D cube
def draw_cube():
    glBegin(GL_QUADS)
    glColor3f(1, 0, 0)  # Red
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, -1, -1)

    glColor3f(0, 1, 0)  # Green
    glVertex3f(-1, -1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, -1, 1)

    glColor3f(0, 0, 1)  # Blue
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, -1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)

    glColor3f(1, 1, 0)  # Yellow
    glVertex3f(1, -1, -1)
    glVertex3f(1, -1, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, 1, -1)

    glColor3f(1, 0, 1)  # Magenta
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, -1, 1)
    glVertex3f(1, -1, 1)
    glVertex3f(1, -1, -1)

    glColor3f(0, 1, 1)  # Cyan
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, 1, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, 1, -1)
    glEnd()

# Calculate the distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function to calculate rotation direction based on hand movement
def calculate_rotation(hand_landmarks):
    global previous_wrist_position, previous_index_position, rotation_x, rotation_y, rotation_z, last_movement_time

    # Get the positions of the wrist and index finger
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # If there is a previous wrist and index position, calculate the change in position
    if previous_wrist_position and previous_index_position:
        # Calculate movement vectors for wrist and index tip
        delta_wrist = np.array([wrist.x - previous_wrist_position[0], wrist.y - previous_wrist_position[1], wrist.z - previous_wrist_position[2]])
        delta_index = np.array([index_tip.x - previous_index_position[0], index_tip.y - previous_index_position[1], index_tip.z - previous_index_position[2]])

        # Check if the movement exceeds the defined threshold for wrist and index
        if np.linalg.norm(delta_wrist) > hand_movement_threshold or np.linalg.norm(delta_index) > hand_movement_threshold:
            # Calculate cross product to find direction of rotation (more complex but accurate for 3D)
            cross_product = np.cross(delta_wrist, delta_index)

            # Update the rotation angles based on the cross product direction and magnitude
            rotation_x += cross_product[0] * 50  # Adjust sensitivity here
            rotation_y += cross_product[1] * 50  # Adjust sensitivity here
            rotation_z += cross_product[2] * 50  # Adjust sensitivity here

            # Update last movement time
            last_movement_time = time.time()

    # Update the previous positions for the next frame
    previous_wrist_position = (wrist.x, wrist.y, wrist.z)
    previous_index_position = (index_tip.x, index_tip.y, index_tip.z)

# Check if both hands are not moving for the specified time (movement_timeout)
def check_hands_stopped():
    global last_movement_time

    # If no movement was detected for the last `movement_timeout` seconds, consider hands stopped
    if time.time() - last_movement_time > movement_timeout:
        return True
    return False

# Main function
def main():
    global zoom_factor, rotation_x, rotation_y, rotation_z

    # OpenCV camera initialization
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found!")
        return

    # Create a Pygame window
    pygame.init()
    screen = pygame.display.set_mode((800, 800), pygame.DOUBLEBUF | pygame.OPENGL)
    
    # OpenGL initialization
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, 1, 0.1, 50.0)

    while True:
        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error accessing camera!")
            break

        # Flip the camera frame to align with OpenGL orientation
        frame = cv2.flip(frame, 1)

        # Process the frame for hand detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw hands landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate the rotation based on hand movement
                calculate_rotation(hand_landmarks)

                # Get the positions of the index and thumb tips for zooming
                if len(results.multi_hand_landmarks) > 0:
                    hand = results.multi_hand_landmarks[0]
                    thumb_tip = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    # Zoom control: Calculate distance between thumb and index finger
                    distance = calculate_distance((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))
                    zoom_factor = max(1.0, distance * 10)  # Adjust zoom factor based on distance

        # Check if both hands are not moving, then reduce size
        if check_hands_stopped():
            zoom_factor = zoom_factor / 2  # Halve the zoom factor (reduce object size)

        # Set up OpenGL projection and modelview
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5 * zoom_factor)  # Apply zoom
        glRotatef(rotation_x, 1, 0, 0)  # Rotate on X axis
        glRotatef(rotation_y, 0, 1, 0)  # Rotate on Y axis
        glRotatef(rotation_z, 0, 0, 1)  # Rotate on Z axis

        # Draw the cube
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_cube()

        # Update Pygame window with OpenGL frame
        pygame.display.flip()

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cap.release()
                cv2.destroyAllWindows()
                return
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pygame.quit()
                cap.release()
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()
