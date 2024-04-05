import cv2 as cv
import numpy as np
from util import get_limits

# Color in BGR
cap = cv.VideoCapture(0)

# Define the coordinates and dimensions of the rectangles
box1 = (100, 100, 50, 50)  # Format: (x, y, width, height)
box2 = (400, 400, 50, 50)  # Format: (x, y, width, height)

while True:
    # Take each frame
    _, frame = cap.read()

    # Draw rectangles on the frame
    cv.rectangle(frame, (box1[0], box1[1]), (box1[0] + box1[2], box1[1] + box1[3]), (0, 255, 0), 2)
    cv.rectangle(frame, (box2[0], box2[1]), (box2[0] + box2[2], box2[1] + box2[3]), (0, 255, 0), 2)

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define range of color in HSV
    lower_colour = np.array([0, 0, 225])
    upper_colour = np.array([0, 0, 255])

    # Threshold the HSV image to get only color colors
    colourmask = cv.inRange(hsv, lower_colour, upper_colour)
    colour_regions = cv.bitwise_and(frame, frame, mask=colourmask)
    colour_gray = cv.cvtColor(colour_regions, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(colour_gray, (5, 5), 0)
    # Otsu thresholding
    _, binary_image = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = 8*cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    closed = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)
    dilate = cv.dilate(closed, kernel, iterations=1)

    # find contours
    contours, _ = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Check if any contours are found
    if contours:
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 100:  # Adjust this threshold as needed
                rect = cv.minAreaRect(contour)
                box = cv.boxPoints(rect)
                box = np.intp(box)
                cv.drawContours(frame, [box], 0, (0, 0, 255), 2)


            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw centroid
                cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                centroid_coords = "Centroid Coordinates (%d, %d)" % (cx, cy)

                # Check if centroid is inside either of the rectangles
                if box1[0] <= cx <= box1[0] + box1[2] and box1[1] <= cy <= box1[1] + box1[3]:
                    cv.rectangle(frame, (box1[0], box1[1]), (box1[0] + box1[2], box1[1] + box1[3]), (0, 0, 255), 2)
                if box2[0] <= cx <= box2[0] + box2[2] and box2[1] <= cy <= box2[1] + box2[3]:
                    cv.rectangle(frame, (box2[0], box2[1]), (box2[0] + box2[2], box2[1] + box2[3]), (0, 0, 255), 2)

    cv.imshow('frame', frame)
    cv.imshow('mask', binary_image)
    cv.imshow('mask2', dilate)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
