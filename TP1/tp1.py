import cv2 as cv

from contours import get_contours, get_shape, check_shape, draw_shape
from image_edits import binary_image, deionised_image


def calculate_image(value):
    pass


def show_original_window(frame):
    cv.imshow("original", frame)


def main():
    # iniciamos la capturadora con el nombre cap
    cap = cv.VideoCapture(0)

    cv.namedWindow('original')
    cv.namedWindow('gray')
    cv.createTrackbar('Gray Thresh', 'original', 100, 255, calculate_image)
    cv.createTrackbar('Circle Thresh', 'original', 9, 100, calculate_image)
    cv.createTrackbar('Square Thresh', 'original', 9, 100, calculate_image)
    cv.createTrackbar('Triangle Thresh', 'original', 66, 100, calculate_image)

    squareImg = cv.imread('./public/square.png')
    square = get_shape(squareImg)

    triangleImg = cv.imread('./public/triangle.jpeg')
    triangle = get_shape(triangleImg)

    circleImg = cv.imread('./public/circle.jpeg')
    circle = get_shape(circleImg)

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
                circle_thresh = check_shape(c, circle, cv.getTrackbarPos('Circle Thresh', 'original')/100)
                square_thresh = check_shape(c, square, cv.getTrackbarPos('Square Thresh', 'original')/100)
                triangle_thresh = check_shape(c, triangle, cv.getTrackbarPos('Triangle Thresh', 'original')/100)

                max_thresh = max(circle_thresh, square_thresh, triangle_thresh)

                if max_thresh < 0:
                    draw_shape(c, 'Unknown', (0, 0, 255), original_frame)
                elif max_thresh == circle_thresh:
                    draw_shape(c, 'Circle', (0, 255, 0), original_frame)
                elif max_thresh == triangle_thresh:
                    draw_shape(c, 'Triangle', (0, 255, 0), original_frame)
                elif max_thresh == square_thresh:
                    draw_shape(c, 'Square', (0, 255, 0), original_frame)
                else:
                    draw_shape(c, 'Unknown', (0, 0, 255), original_frame)

        show_original_window(original_frame)

        if cv.waitKey(10) == ord('z'):
            break

    # apagamos la capturadora y cerramos las ventanas que se nos abrieron
    cap.release()


main()
cv.destroyAllWindows()
