import cv2
import numpy as np
import os
from datetime import datetime

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # PiCamera kullanılacak
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

barcodeDetected = 0


def enlarge_box(box, scale=20):
    # Calculate the center of the box
    center = np.mean(box, axis=0)

    # Move each point away from the center by the scale
    new_box = np.zeros_like(box)
    for i in range(4):
        new_box[i] = box[i] + (box[i] - center) * scale / np.linalg.norm(
            box[i] - center
        )

    return new_box


def get_rotated_roi(img, box):
    # Order points in the correct order
    rect = cv2.minAreaRect(box)
    box = cv2.boxPoints(rect).astype(int)

    # Determine the ordering of the points and order them
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = cv2.minAreaRect(box)
    box = sorted(list(box), key=lambda x: x[1])
    top = sorted(box[:2], key=lambda x: x[0])
    bottom = sorted(box[2:], key=lambda x: x[0], reverse=True)
    box = np.array(top + bottom, dtype="float32")

    # Compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.linalg.norm(box[2] - box[3])
    widthB = np.linalg.norm(box[1] - box[0])
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.linalg.norm(box[1] - box[2])
    heightB = np.linalg.norm(box[0] - box[3])
    maxHeight = max(int(heightA), int(heightB))

    # Now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # Return the warped image
    return warped


while True:
    success, img = cap.read()

    # Görüntüyü gri tonlamaya dönüştür
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, kernel, iterations=4)
    closed = cv2.dilate(closed, kernel, iterations=4)

    cv2.imshow("Threshold", closed)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    (cnts, _) = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    large_contours = [c for c in cnts if cv2.contourArea(c) > 5000]

    if len(cnts) > 0:

        for c in large_contours:

            c = max(cnts, key=cv2.contourArea)  # Get the largest contour
            rect = cv2.minAreaRect(c)  # Get minimum area rectangle
            box = cv2.boxPoints(rect).astype(int)  # Get box points

            enlarged_box = enlarge_box(box)
            # Convert points to int
            cv2.drawContours(img, [enlarged_box], -1, (0, 255, 0), 1)

            rotated_roi = get_rotated_roi(img, enlarged_box)

            now = datetime.now()
            datetime_suffix = now.strftime("%Y%m%d_%H%M%S")
            folder_suffix = now.strftime("%Y%m%d_%H%M")
            output_folder = "outputs/" + folder_suffix

            if barcodeDetected < 5:
                # Ensure the outputs folder exists
                os.makedirs(output_folder, exist_ok=True)
                filename = f"detected_barcode_{datetime_suffix}.jpg"
                filepath = os.path.join(output_folder, filename)

                cv2.imwrite(filepath, rotated_roi)
                barcodeDetected = barcodeDetected + 1

    cv2.imshow("Gradient", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Break the loop if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
