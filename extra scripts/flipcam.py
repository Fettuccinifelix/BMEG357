import cv2

# Open the video camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret:
        print("Error: Couldn't capture frame")
        break

    # Flip the frame horizontally
    mirrored_frame = cv2.flip(frame, 1)

    # Display the mirrored frame
    cv2.imshow('Mirrored Output', mirrored_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()