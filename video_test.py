import picamera

camera = picamera.PiCamera()

camera.resolution = (640, 480)
camera.start_recording('test_video.h264')
camera.wait_recording(5)
camera.stop_recording()

print('Finished recording')

# önemli paketlerin eklenmesi
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

# kamerayı başlat ve ham kamera yakalaması için bir referans al
camera = PiCamera()
rawCapture = PiRGBArray(camera)
# kameranın yüklenmesi için bekleme süresi
time.sleep(0.1)
# kameradan görüntü yakalama işlemi
camera.capture(rawCapture, format="bgr")
image = rawCapture.array
# ekrana yakalanan görüntünün basılması
cv2.imshow("Image", image)
cv2.waitKey(0)