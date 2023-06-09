import cv2
import numpy as np
import imutils

color_ranges = {
    "red": (np.array([0, 50, 50]), np.array([10, 255, 255])),
    "blue": (np.array([100, 50, 50]), np.array([130, 255, 255])),
    "green": (np.array([50, 50, 50]), np.array([80, 255, 255])),
    "yellow": (np.array([20, 50, 50]), np.array([40, 255, 255])),
    "orange": (np.array([10, 50, 50]), np.array([20, 255, 255])),
    "purple": (np.array([130, 50, 50]), np.array([170, 255, 255])),
    "pink": (np.array([170, 50, 50]), np.array([180, 255, 255])),
    "black": (np.array([0, 0, 0]), np.array([180, 255, 50])),
    "white": (np.array([0, 0, 200]), np.array([180, 50, 255])),
    "brown": (np.array([10, 50, 50]), np.array([30, 255, 150])),
    "gray": (np.array([0, 0, 50]), np.array([180, 50, 200])),
    "beige": (np.array([30, 50, 100]), np.array([60, 150, 255])),
    "turquoise": (np.array([80, 50, 50]), np.array([100, 255, 255])),
    "gold": (np.array([20, 100, 50]), np.array([30, 255, 255])),
    "silver": (np.array([0, 0, 100]), np.array([180, 25, 200])),
    "indigo": (np.array([110, 50, 50]), np.array([130, 255, 255])),
    "lavender": (np.array([130, 50, 50]), np.array([150, 255, 255])),
    "maroon": (np.array([0, 50, 50]), np.array([10, 255, 150])),
    "teal": (np.array([80, 50, 50]), np.array([100, 255, 150])),
    "fuchsia": (np.array([150, 50, 50]), np.array([170, 255, 255])),
    "skin": (np.array([0, 20, 70]), np.array([20, 255, 255]))

}


# Define a function to calculate the distance
def calculate_distance(focal_length, actual_width, pixel_width):
     return (actual_width * focal_length) / pixel_width

# Define the actual width of the object (in this case, a green piece of paper)
actual_width = 8.5 # centimeters

# Define the focal length of the camera (you can calibrate this by taking a picture of a known object and measuring its actual width)
focal_length = 875.0

# Initialize the video stream
cap = cv2.VideoCapture(0)

# Define the flag to keep track of whether a color has been detected
color_detected = False

while True:
    # Prompt the user to input the color to detect
    color = input(''' Available colors {"red","blue","green","yellow","orange","purple","pink","black","white","brown","gray","gold","silver","beige","turquoise","indigo","skin","lavender","maroon","fuchsia","teal"}
    Enter the color to detect (or 'q' to quit) : ''')
    if color == 'q':
        break

    # Check if the color is valid
    if color not in color_ranges:
        print("Invalid color")
        continue

    color_detected = False

    # Continuously detect the specified color
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        # Resize the frame
        frame = imutils.resize(frame, width=600)

        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the image to get only the color
        lower_color, upper_color = color_ranges[color]
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Find contours in the mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a bounding box around the largest contour
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Calculate the distance to the object
            pixel_width = w
            distance = calculate_distance(focal_length, actual_width, pixel_width)
            color_detected = True

        # Display the distance on the frame if a color is detected
        if color_detected:
            cv2.putText(frame, "{} color distance :{:.2f} cm".format(color.capitalize(), distance), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Welcome In Distance Measurement System", frame)

        # Check if the user pressed the 'q' key to quit or 'c' key to change the color
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            break


# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
