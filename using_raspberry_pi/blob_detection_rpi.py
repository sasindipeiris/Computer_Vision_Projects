import cv2
import numpy as np

# Open the USB camera (0 is the default camera)
video_capture = cv2.VideoCapture(0)

while True:
    result, video_frame = video_capture.read()  # Read frames from the video
    if not result:
        break  # Exit the loop if no frame is captured

    # Convert the frame to HSV (Hue, Saturation, Value) color space
    hsv_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color orange in HSV
    lower_orange = np.array([10, 100, 100])   # Lower bound (Hue, Saturation, Value)
    upper_orange = np.array([25, 255, 255])   # Upper bound

    # Create a mask that keeps only the orange parts of the image
    mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

    # Apply bitwise AND to extract the orange color from the original frame
    orange_detected = cv2.bitwise_and(video_frame, video_frame, mask=mask)

    # Find contours (shapes) of the detected orange blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected orange blobs
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignore small detections (noise)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

    # Show the original frame with bounding boxes
    cv2.imshow("Orange Color Blob Detection", video_frame)

    # Show the mask (only for visualization)
    cv2.imshow("Mask", mask)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close windows
video_capture.release()
cv2.destroyAllWindows()
