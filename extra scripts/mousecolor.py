import cv2 as cv
import numpy as np

# Initialize global variables to store mouse position and average HSV value
mouseX, mouseY = -1, -1
avg_hsv = None
box_size = 20  # Initial box size

# Mouse callback function to update mouse position
def update_mouse_pos(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv.EVENT_MOUSEMOVE:
        mouseX, mouseY = x, y

# Function to compute the average HSV value of the region around the mouse cursor
def compute_avg_hsv(frame, x, y, box_size):
    # Extract the region around the mouse cursor
    region = frame[max(0, y - box_size):min(frame.shape[0], y + box_size),
                   max(0, x - box_size):min(frame.shape[1], x + box_size)]

    # Convert the region to HSV color space
    hsv_region = cv.cvtColor(region, cv.COLOR_BGR2HSV)

    # Compute the average HSV value
    avg_h = int(np.mean(hsv_region[:, :, 0]))
    avg_s = int(np.mean(hsv_region[:, :, 1]))
    avg_v = int(np.mean(hsv_region[:, :, 2]))

    return (avg_h, avg_s, avg_v)

# Start capturing from the webcam
cap = cv.VideoCapture(0)

# Create a window and set the mouse callback function
cv.namedWindow('frame')
cv.setMouseCallback('frame', update_mouse_pos)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Draw a rectangle around the mouse cursor
    cv.rectangle(frame, (max(0, mouseX - box_size), max(0, mouseY - box_size)),
                 (min(frame.shape[1], mouseX + box_size), min(frame.shape[0], mouseY + box_size)),
                 (0, 255, 0), 2)

    # Compute the average HSV value of the region around the mouse cursor
    avg_hsv = compute_avg_hsv(frame, mouseX, mouseY, box_size)

    # Display the average HSV value
    if avg_hsv is not None:
        cv.putText(frame, f'Avg HSV: {avg_hsv}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv.imshow('frame', frame)

    # Check for key presses
    key = cv.waitKey(1)
    if key == ord('q'):  # Quit the loop if 'q' is pressed
        break
    elif key == 63232:  # Increase box size when 'up' arrow key is pressed
        box_size += 5
    elif key == 63233 and box_size > 5:  # Decrease box size when 'down' arrow key is pressed (minimum size is 5)
        box_size -= 5

# Release the capture and close all windows
cap.release()
cv.destroyAllWindows()
