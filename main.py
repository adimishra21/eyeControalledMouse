import cv2
import numpy as np
import pyautogui

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
if eye_cascade.empty():
    print("Error loading Haar Cascade file. Please check the file path.")

# Function to calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    if len(eye) < 6:
        return 0  # Return a default value if not enough points
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds for blink detection
EAR_THRESHOLD = 0.2
BLINK_COUNTER = 0
BLINK_LIMIT = 3  # Number of frames to consider a blink

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    # If eyes are detected, control the mouse
    if len(eyes) > 0:
        # Get the coordinates of the first detected eye
        (x, y, w, h) = eyes[0]
        eye_center_x = x + w // 2
        eye_center_y = y + h // 2

        # Map the eye position to the screen size
        screen_width, screen_height = pyautogui.size()
        mouse_x = np.interp(eye_center_x, [0, frame.shape[1]], [0, screen_width])
        mouse_y = np.interp(eye_center_y, [0, frame.shape[0]], [0, screen_height])

        # Move the mouse
        pyautogui.moveTo(mouse_x, mouse_y)

        # Eye aspect ratio for blink detection
        ear = eye_aspect_ratio([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x + w // 2, y + h // 2)])

        # Check for blink
        if ear < EAR_THRESHOLD:
            BLINK_COUNTER += 1
            if BLINK_COUNTER >= BLINK_LIMIT:
                pyautogui.click()  # Perform mouse click
                BLINK_COUNTER = 0  # Reset counter
        else:
            BLINK_COUNTER = 0  # Reset counter if eyes are open

    # Display the resulting frame
    cv2.imshow('Eye Controlled Mouse', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
