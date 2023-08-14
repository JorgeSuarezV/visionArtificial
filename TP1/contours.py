from random import random

import cv2 as cv
from image_edits import binary_image, deionised_image


def get_contours(thresh):
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv.contourArea(c) > 1000]
    if len(contours) > 0:
        contours.pop(0)
    return contours


def get_shape(shape):
    binary = binary_image(shape, 120)
    d_image = deionised_image(binary)
    conts = get_contours(d_image)
    if len(conts) > 0:
        cv.drawContours(binary, conts, -1, (255, 255, 0), 3)
        cv.imshow(f"{shape}", binary)
        return conts[0]


def check_shape(contour, image, threshold):
    return cv.matchShapes(contour, image, cv.CONTOURS_MATCH_I2, 0) < threshold


def draw_shape(contour, name, color, image):
    cv.drawContours(image, contour, -1, color, 3)
    x, y, w, h = cv.boundingRect(contour)
    cv.putText(image, name, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 1, color, 1,
               cv.LINE_AA)
