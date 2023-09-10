import cv2
import numpy as np 

# Function to calculate Hu Moments
def calculate_hu_moments(contour):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)
    return [-np.sign(moment) * np.log10(np.abs(moment)) for moment in hu_moments]

def generate_descriptors():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Initialize lists to store descriptors and labels
    descriptors = []
    sample_labels = []

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
                hu_moments = calculate_hu_moments(contour)
                
                # Display Hu Moments in the console
                print("Hu Moments:", hu_moments)
                
                # Ask the user to input the label
                label = int(input("Enter the label (1: Square, 2: Triangle, 3: Star): "))
                
                # Store the descriptor and label
                descriptors.append(hu_moments)
                sample_labels.append(label)

        # Display the real-time image
        cv2.imshow("Descriptor Generator", frame)

        # Exit the process with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
