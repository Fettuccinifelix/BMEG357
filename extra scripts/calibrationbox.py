import cv2 as cv
import numpy as np
from util import get_limits

# Color in BGR
colour = [255, 255, 255]
cap = cv.VideoCapture(0)

# Define the coordinates and dimensions of the calibration box
calibration_box = (300, 200, 15, 15)  # Format: (x, y, width, height)

# Initialize variable to store selected color
selected_color = None

def print_hsv_values(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        # Get HSV values of the color in the calibration box
        hsv_color = cv.cvtColor(np.uint8([[frame[y, x]]]), cv.COLOR_BGR2HSV)[0][0]
        print("HSV Values:", hsv_color)
        # Save the color for later use
        global selected_color
        selected_color = hsv_color

while True:
    # Take each frame
    _, frame = cap.read()

    # Draw calibration box on the frame
    cv.rectangle(frame, (calibration_box[0], calibration_box[1]),
                 (calibration_box[0] + calibration_box[2], calibration_box[1] + calibration_box[3]),
                 (0, 255, 0), 2)

    # Show frame
    cv.imshow('frame', frame)

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Add mouse click event to get HSV values
    cv.setMouseCallback('frame', print_hsv_values)

    # define range of colour color in HSV
    lower_colour, upper_colour = get_limits(color=colour)

    # Threshold the HSV image to get only color colors
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

    cv.imshow('mask', binary_image)
    cv.imshow('mask2', dilate)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
