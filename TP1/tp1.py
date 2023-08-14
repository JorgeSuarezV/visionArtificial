import cv2 as cv

from contours import get_contours


def calculate_image(value):
    pass


def show_original_window(frame):
    cv.imshow("original", frame)


def binary_image(original_frame, trackbar_value):
    gray = cv.cvtColor(original_frame, cv.COLOR_RGB2GRAY)
    _, thresh1 = cv.threshold(gray, trackbar_value, 255, cv.THRESH_BINARY)
    return thresh1


def deionised_image(image):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)


def main():
    # iniciamos la capturadora con el nombre cap
    cap = cv.VideoCapture(0)

    cv.namedWindow('original')
    cv.namedWindow('gray')
    cv.createTrackbar('Gray Thresh', 'original', 0, 255, calculate_image)
    # cv.createTrackbar('Contours thresh', 'original', 10, 255, calculate_contours)

    while True:
        _, original_frame = cap.read()
        original_frame = cv.flip(original_frame,1)

        trackbar_value = cv.getTrackbarPos('Gray Thresh', 'original')
        bg_image = binary_image(original_frame, trackbar_value)
        cv.imshow("gray", bg_image)

        d_image = deionised_image(bg_image)

        get_contours(d_image, original_frame)
        show_original_window(original_frame)

        if cv.waitKey(10) == ord('z'):
            break

    # apagamos la capturadora y cerramos las ventanas que se nos abrieron
    cap.release()


main()
cv.destroyAllWindows()
