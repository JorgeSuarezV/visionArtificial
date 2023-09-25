import cv2 as cv
import numpy as np
from contours import get_contours, check_shape, draw_shape
from image_edits import binary_image, deionised_image
from joblib import load


def calculate_image(value):
    pass


def show_original_window(frame):
    cv.imshow("original", frame)


def main():
    # Load the trained decision tree classifier
    clasificadorRecuperado = load('classifier.joblib')

    # Initialize the video capture
    cap = cv.VideoCapture(0)

    cv.namedWindow('original')
    cv.createTrackbar('Gray Thresh', 'original', 100, 255, calculate_image)


    while True:
        _, original_frame = cap.read()
        original_frame = cv.flip(original_frame, 1)

        trackbar_value = cv.getTrackbarPos('Gray Thresh', 'original')
        bg_image = binary_image(original_frame, trackbar_value)
        cv.imshow("gray", bg_image)

        d_image = deionised_image(bg_image)

        conts = get_contours(d_image)
        if len(conts) > 0:
            for c in conts:
                hu_moments = cv.HuMoments(cv.moments(c)).flatten()

                # Predict the shape using the trained model
                shape_tag = clasificadorRecuperado.predict([hu_moments])

                if shape_tag == 1:
                    shape_name = '5-point-star'
                    color = (0, 255, 0)
                elif shape_tag == 2:
                    shape_name = 'Rectangle'
                    color = (0, 255, 0)
                elif shape_tag == 3:
                    shape_name = 'Triangle'
                    color = (0, 255, 0)
                else:
                    shape_name = 'Unknown'
                    color = (0, 0, 255)

                draw_shape(c, shape_name, color, original_frame)

        show_original_window(original_frame)

        if cv.waitKey(10) == ord('z'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
