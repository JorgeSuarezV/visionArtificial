import cv2 as cv

# iniciamos la capturadora con el nombre cap
cap = cv.VideoCapture(0)

cv.namedWindow('img1')

last_value = 0


def calculate_image(val):
    global last_value
    last_value = val
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, val, 255, cv.THRESH_BINARY)
    cv.imshow("img1", thresh1)


cv.createTrackbar('Trackbar', 'img1', 0, 255, calculate_image)

while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    calculate_image(last_value)

    # al presionar z salimos del loop
    if cv.waitKey(10) == ord('z'):
        break

# apagamos la capturadora y cerramos las ventanas que se nos abrieron
cap.release()
cv.destroyAllWindows()
