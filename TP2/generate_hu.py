import cv2
import numpy as np
import os
import csv


def get_contours(frame, mode, method):
    contours, hierarchy = cv2.findContours(frame, mode, method)
    return contours


def get_biggest_contour(contours):
    max_cnt = contours[0]
    for cnt in contours:
        if cv2.contourArea(cnt) > cv2.contourArea(max_cnt):
            max_cnt = cnt
    return max_cnt


def calculate_hu_moments(contour):
    return cv2.HuMoments(cv2.moments(contour)).flatten()


def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary mask
    _, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)

    # Perform morphological operations to remove small white regions (dots)
    kernel = np.ones((7, 7), np.uint8)
    cleaned_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return cleaned_mask


def save_moment_to_csv(hu_moments, file_name, tag):
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([tag] + list(hu_moments))


def get_files_from_folder(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


def main():
    csv_filename = 'hu_moments.csv'
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Tag', 'HuMoment1', 'HuMoment2', 'HuMoment3', 'HuMoment4', 'HuMoment5', 'HuMoment6', 'HuMoment7'])

    shape_folders = ['5-point-star', 'rectangle', 'triangle']

    for tag, folder in enumerate(shape_folders, start=1):
        files_in_folder = get_files_from_folder(f'./shapes/{folder}')

        for filename in files_in_folder:
            image = cv2.imread(f'./shapes/{folder}/{filename}')

            # Preprocess the image to remove small white dots
            cleaned_mask = preprocess_image(image)

            if folder.startswith('circle'):
                cv2.imshow('cleaned', cleaned_mask)
                cv2.waitKey(0)

            # Find contours in the cleaned mask
            contours = get_contours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                biggest_contour = get_biggest_contour(contours)
                hu_moments = calculate_hu_moments(biggest_contour)
                save_moment_to_csv(hu_moments, csv_filename, tag)


if __name__ == "__main__":
    main()
