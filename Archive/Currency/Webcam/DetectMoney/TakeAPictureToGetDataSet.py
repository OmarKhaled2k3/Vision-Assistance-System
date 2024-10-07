# import the necessary packages
import time
import cv2
import imutils

cam = cv2.VideoCapture(1)
cam.set(3, 320) # set video width
cam.set(4, 240) # set video height
time.sleep(2) #delay 2sec let camera stable
ret, frame = cam.read()
# Cap a picture and save to /home/pi/DetectMoney/TrainingData with name 20000B.img
cv2.imwrite("TrainingData/5F.jpg",frame) 
