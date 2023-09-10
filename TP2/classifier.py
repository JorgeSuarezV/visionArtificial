import cv2
from joblib import load
import numpy as np
import LABELS from labels

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the image to find contours
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) > 1000:
            # Calculate Hu Moments
            hu_moments = calculate_hu_moments(contour)  # You can reuse the function from the descriptor generator
            
            # Make the prediction
            predicted_label = classifier.predict([hu_moments])[0]
            
            # Get the label description
            label_description = LABELS.get(predicted_label, "Unknown")
            
            # Display the label on the frame
            cv2.putText(frame, label_description, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the real-time image
    cv2.imshow("Object Classifier", frame)

    # Exit the process with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
