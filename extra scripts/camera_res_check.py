import cv2 as cv

# Open the camera
cap = cv.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Read a frame from the camera
ret, frame = cap.read()

# Check if the frame is read successfully
if not ret:
    print("Error: Could not read frame.")
    exit()

# Get the size of the frame
height, width, _ = frame.shape

print("Frame size: {}x{}".format(width, height))

# Release the camera
cap.release()