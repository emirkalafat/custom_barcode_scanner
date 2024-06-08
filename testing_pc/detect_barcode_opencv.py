# import the necessary packages
import numpy as np
import cv2
from cv2.typing import MatLike

from barcode_detect_test import detect_barcode

show = 1

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # PiCamera kullanılacak
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


# first a conservative filter for grayscale images will be defined.
def conservative_smoothing_gray(data, filter_size):
    temp = []
    indexer = filter_size // 2
    new_image = data.copy()
    nrow, ncol = data.shape
    for i in range(nrow):
        for j in range(ncol):
            for k in range(i - indexer, i + indexer + 1):
                for m in range(j - indexer, j + indexer + 1):
                    if (k > -1) and (k < nrow):
                        if (m > -1) and (m < ncol):
                            temp.append(data[k, m])
            temp.remove(data[i, j])
            max_value = max(temp)
            min_value = min(temp)
            if data[i, j] > max_value:
                new_image[i, j] = max_value
            elif data[i, j] < min_value:
                new_image[i, j] = min_value
            temp = []
    return new_image.copy()


def enlarge_box(box, scale=10):
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


def capture_barcode_box(c: MatLike):
    rect = cv2.minAreaRect(c)  # Get minimum area rectangle
    box = cv2.boxPoints(rect).astype(int)  # Get box points
    enlarged_box = enlarge_box(box)
    # Convert points to int
    cv2.drawContours(img, [enlarged_box], -1, (100, 255, 100), 1)

    rotated_roi = get_rotated_roi(img, enlarged_box)

    return rotated_roi


while True:
    success, img = cap.read()

    # load the image and convert it to grayscale
    image = img

    # resize image
    image = cv2.resize(image, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # calculate x & y gradient
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    if show == 1:
        cv2.imshow(
            "gradient-sub",
            cv2.resize(gradient, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC),
        )

    # blur the image
    blurred = cv2.blur(gradient, (3, 3))

    # threshold the image
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    if show == 1:
        cv2.imshow(
            "threshed",
            cv2.resize(thresh, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC),
        )

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    if show == 1:
        cv2.imshow(
            "morphology",
            cv2.resize(closed, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC),
        )

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)  # type: ignore
    closed = cv2.dilate(closed, None, iterations=4)  # type: ignore

    if show == 1:
        cv2.imshow(
            "erode/dilate",
            cv2.resize(closed, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC),
        )

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts, hierarchy = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2:]

    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # c1 = sorted(cnts, key = cv2.contourArea, reverse = True)[1]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(int)
    # rect1 = cv2.minAreaRect(c1)
    # box1 = cv2.boxPoints(rect1).astype(int)

    # draw a bounding box arounded the detected barcode and display the
    # image

    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    # cv2.drawContours(image, [box1], -1, (0, 255, 0), 3)

    cv2.putText(
        image,
        "Barcode",
        (box[2][0], box[2][1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
    image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    cv2.imshow("Image", image)

    # sleep(0.1)

    capture = cv2.waitKey(1) & 0xFF == ord("x")

    if capture:
        (cnts, _) = cv2.findContours(
            closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        c = max(cnts, key=cv2.contourArea)

        barcode = capture_barcode_box(c)
        cv2.imwrite("captured_barcode.jpg", barcode)

        detect_barcode("captured_barcode.jpg")

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Break the loop if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
