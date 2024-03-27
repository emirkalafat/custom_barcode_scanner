import cv2
import numpy as np
import cvzone
from pyzbar.pyzbar import decode
import os
from datetime import datetime

cap = cv2.VideoCapture(0)  # PiCamera kullanılacak

barcodeDetected = False

# Ensure the outputs folder exists
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

while True:
    success, img = cap.read()

    if not success:
        break

    for code in decode(img):
        decoded_data = code.data.decode("utf-8")

        rect_pts = code.rect

        if decoded_data and not barcodeDetected:

            barcodeDetected = True

            now = datetime.now()
            datetime_suffix = now.strftime("%Y%m%d_%H%M%S")
            filename = f"detected_barcode_{datetime_suffix}.jpg"
            filepath = os.path.join(output_folder, filename)

            pts = np.array([code.polygon], np.int32)
            cv2.polylines(img, [pts], True, (255, 0, 0), 1)
            cv2.putText(
                img,
                "QR Code",
                (rect_pts[0], rect_pts[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )

            cv2.imwrite(filepath, img)
            print("Barcode detected and image saved!")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Break the loop if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
