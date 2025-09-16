import cv2
import numpy as np

# Function to calculate the Manhattan distance between two points
def calculate_distance(p1, p2):
    return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])

# Initialize the webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    # Define the range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the HSV image to get only black colors
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Threshold the HSV image to get only yellow colors
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours for black
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    black_point = None

    # Find the largest black contour and use it to determine the black point (reference)
    if contours_black:
        largest_contour_black = max(contours_black, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(largest_contour_black)
        black_point = (x + w // 2, y + h // 2)

        # Draw a circle at the black point
        cv2.circle(frame, black_point, 5, (0, 255, 0), -1)


    # Find contours for yellow
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yellow_point = None

    # Find the largest yellow contour and use it to determine the yellow point (dynamic)
    if contours_yellow:
        largest_contour_yellow = max(contours_yellow, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(largest_contour_yellow)
        yellow_point = (x + w // 2, y + h // 2)

        # Draw a circle at the yellow point
        cv2.circle(frame, yellow_point, 5, (0, 255, 255), -1)

   # Calculate and display distance
    if black_point and yellow_point:
        distance_pixels = calculate_distance(black_point, yellow_point)
        distance_cm = distance_pixels*(11/71)  # Convert to cm
        cv2.putText(frame, f"Distance: {distance_cm:.2f} cm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
