
import cv2 as cv
import numpy as np




cap = cv.VideoCapture(0)

while 1:

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    bluemask = cv.inRange(hsv, lower_blue, upper_blue)
    bluemask_channels = cv.split(bluemask)


    #Correct for nonuniform illumination
    greymask = bluemask_channels[0]
    se = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))
    bg = cv.morphologyEx(greymask, cv.MORPH_DILATE, se)
    out_grey = cv.divide(greymask, bg, scale=255)
    out_binary = cv.threshold(out_grey, 0, 255, cv.THRESH_OTSU)[1]


    # find contours
    contours = cv.findContours(greymask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]
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


    hull_list = []
    for i in range(len(contours)):
        hull = cv.convexHull(contours[i])
        hull_list.append(hull)

    for h in hull_list:
        cv.drawContours(frame, [h], -1, (0, 255, 0), 3)



    cv.imshow('frame', frame)
    cv.imshow('mask', out_binary)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()