import cv2 as cv
import numpy as np
from util import get_limits


# color in bgr

colour = [255, 0, 0]
kernel = 100 * np.ones((5, 5), np.uint8)
cap = cv.VideoCapture(0)

box1 = (100, 100)
box2 = (400, 400)

while 1:

    # Take each frame
    _, frame = cap.read()

    frame = cv.flip(frame, 1)

    cv.rectangle(frame, (box1[0] - 25, box1[1] - 25), (box1[0] + 25, box1[1] + 25), (255, 0, 0),2)  # Format: (top left), (bottom right)
    cv.rectangle(frame, (box2[0] - 25, box2[1] - 25), (box2[0] + 25, box2[1] + 25), (0, 255, 0),2)  # Format: (top left), (bottom right)

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

    ''' 
            #MULTI-INPUT SENSING
            if area > 200:
                rect = cv.minAreaRect(contour)
                box = cv.boxPoints(rect)
                box = np.intp(box)
                cv.drawContours(frame, [box], 0, (0, 0, 255), 2)
    '''
    '''
    for c in contours:
        # calculate moments for each contour
        M = cv.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
    
    #for c in contours:
    #    x, y, w, h = cv.boundingRect(c)
    #    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #    cv.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    
    if contours:
        for contour in contours:
            # Check if the contour is valid (contains at least 3 points)
            if len(contour) >= 3:
                # Calculate convex hull
                rect = cv.minAreaRect(contour)
                box = cv.boxPoints(rect)
                box = np.intp(box)
                cv.drawContours(frame, [box], 0, (0, 0, 255), 2)
    '''

    cv.imshow('frame', frame)
    cv.imshow('mask', binary_image)
    cv.imshow('mask2', dilate)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
