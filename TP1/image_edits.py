import cv2 as cv


def binary_image(original_frame, trackbar_value):
    gray = cv.cvtColor(original_frame, cv.COLOR_RGB2GRAY)
    _, thresh1 = cv.threshold(gray, trackbar_value, 255, cv.THRESH_BINARY)
    return thresh1


def deionised_image(image):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

