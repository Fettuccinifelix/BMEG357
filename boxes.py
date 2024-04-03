import cv2 as cv
import numpy as np
from util import get_limits
import random

# Function to generate a new position for box2 if it overlaps with box1
def generate_new_position():
    while True:
        x = random.randint(50, frame_width - 50)
        y = random.randint(50, frame_height - 50)
        # Check if the new position overlaps with the reset box
        if not (frame_height - 50 <= y <= frame_height and frame_width - 100 <= x <= frame_width):
            return x, y

# Color in BGR
colour = [255, 0, 0]
cap = cv.VideoCapture(0)

# Get frame dimensions
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Generate initial random positions for the rectangles
box1 = generate_new_position()
box2 = generate_new_position()

centroid_in_box1 = False
centroid_in_box2 = False
centroid_in_reset_box = False

instruction = "Put the centroid in the magenta box"

while True:
    # Take each frame
    _, frame = cap.read()
    frame = cv.flip(frame, 1)

    cv.rectangle(frame, (box1[0] - 25, box1[1] - 25), (box1[0] + 25, box1[1] + 25), (255, 0, 255),
                 2)  # Format: (top left), (bottom right)
    cv.rectangle(frame, (box2[0] - 25, box2[1] - 25), (box2[0] + 25, box2[1] + 25), (255, 255, 0),
                 2)  # Format: (top left), (bottom right)

    cv.putText(frame, instruction, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of colour color in HSV
    lower_colour, upper_colour = get_limits(color=colour)

    # Threshold the HSV image to get only colour colors
    colourmask = cv.inRange(hsv, lower_colour, upper_colour)
    colour_regions = cv.bitwise_and(frame, frame, mask=colourmask)
    colour_gray = cv.cvtColor(colour_regions, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(colour_gray, (5, 5), 0)
    # Otsu thresholding
    _, binary_image = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    closed = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)
    dilate = cv.dilate(closed, kernel, iterations=1)

    # find contours
    contours, _ = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Check if any contours are found
    if contours:

        # Iterate through each contour to find valid
        biggest_area = -1
        biggest_contour = None
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 300:  # Adjust this threshold as needed
                if area > biggest_area:
                    biggest_area = area
                    biggest_contour = contour

        if biggest_contour is not None:
            rect = cv.minAreaRect(biggest_contour)
            box = cv.boxPoints(rect)
            box = np.intp(box)
            cv.drawContours(frame, [box], 0, (0, 0, 255), 2)

            M = cv.moments(biggest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                print("Centroid Coordinates (x, y):", cx, cy)

                # Draw centroid
                cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                centroid_coords = "Centroid Coordinates (%d, %d)" % (cx, cy)
                cv.putText(frame, centroid_coords, (cx - 25, cy - 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Check if centroid is inside the boxes
                if not centroid_in_box1 and box1[0] - 25 <= cx <= box1[0] + 25 and box1[1] - 25 <= cy <= box1[1] + 25:
                    if instruction == "Put the centroid in the magenta box":
                        centroid_in_box1 = True
                        instruction = "Put the centroid in the cyan box"
                elif not centroid_in_box2 and box2[0] - 25 <= cx <= box2[0] + 25 and box2[1] - 25 <= cy <= box2[1] + 25:
                    if instruction == "Put the centroid in the cyan box":
                        centroid_in_box2 = True
                        instruction = "Congratulations! You've placed the centroid correctly."
                elif not centroid_in_reset_box and frame_height - 50 <= cy <= frame_height and frame_width - 100 <= cx <= frame_width:
                    # Reset boxes and instruction
                    box1 = generate_new_position()
                    box2 = generate_new_position()
                    centroid_in_box1 = False
                    centroid_in_box2 = False
                    instruction = "Put the centroid in the magenta box"
                    centroid_in_reset_box = True


    # Reset centroid_in_reset_box flag if the centroid moves out of the reset box
    if centroid_in_reset_box and not (frame_height - 50 <= cy <= frame_height and frame_width - 100 <= cx <= frame_width):
        centroid_in_reset_box = False

    # Change color of the boxes based on centroid positions
    if centroid_in_box1:
        cv.rectangle(frame, (box1[0] - 25, box1[1] - 25), (box1[0] + 25, box1[1] + 25), (0, 0, 255), 2)
    if centroid_in_box2:
        cv.rectangle(frame, (box2[0] - 25, box2[1] - 25), (box2[0] + 25, box2[1] + 25), (0, 0, 255), 2)

    # Add a reset box at the bottom right
    cv.rectangle(frame, (frame_width - 100, frame_height - 50), (frame_width, frame_height),
                 (255, 255, 255), -1)
    cv.putText(frame, "Reset", (frame_width - 150, frame_height - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
               2)

    cv.imshow('frame', frame)
    cv.imshow('mask', binary_image)
    cv.imshow('mask2', dilate)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
