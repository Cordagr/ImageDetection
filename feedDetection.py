import cv2
import numpy as np

def ORB_detector(new_image, image_template):
    # Convert to grayscale
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Create ORB detector with 300 keypoints (reducing for better precision)
    orb = cv2.ORB_create(300, 1.2)

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image_template, None)

    # Check if descriptors are valid (prevents crashing)
    if des1 is None or des2 is None:
        return 0  # No matches found

    # Use BFMatcher without crossCheck
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe’s Ratio Test (remove weak matches)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    return len(good_matches)

# Start webcam
cap = cv2.VideoCapture(0)

# Load reference image
image_template = cv2.imread('banana.jpg', 0) 

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break  # If frame is not captured, exit loop

    # Get dimensions
    height, width = frame.shape[:2]

    # Define Region of Interest (ROI)
    top_left_x = int(width / 3)
    top_left_y = int(height / 3)
    bottom_right_x = int((width / 3) * 2)
    bottom_right_y = int((height / 3) * 2)

    # Draw ROI Box
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 255, 255), 2)

    # Corrected ROI cropping
    cropped = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Flip frame horizontally (optional)
    frame = cv2.flip(frame, 1)

    # Get number of ORB matches
    matches = ORB_detector(cropped, image_template)

    # Display match count
    output_string = f"Matches = {matches}"
    cv2.putText(frame, output_string, (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 0, 150), 2)

    # Object detection threshold (increase if detecting face instead of object)
    threshold = 400  

    if matches > threshold:
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 3)
        cv2.putText(frame, 'Object Found', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Show output frame
    cv2.imshow('Object Detector using ORB', frame)
    if cv2.waitKey(1) == 13:  # Press 'Enter' to exit
        break

cap.release()
cv2.destroyAllWindows()
