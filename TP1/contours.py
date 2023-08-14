import cv2 as cv


def get_contours(thresh, image):
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv.contourArea(c) > 1000]
    if len(contours) > 0:
        contours.pop(0)
        draw_contours(contours, image)


def draw_contours(contours, image):
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
    cv.drawContours(image, contours, -1, (0, 255, 0), 3)
