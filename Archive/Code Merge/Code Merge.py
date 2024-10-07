import cv2
from easyocr import Reader
import argparse
import time
import numpy as np
import imutils
import pickle
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import pyttsx3
#engine = pyttsx3.init() # object creation

switch=0
camera = cv2.VideoCapture(0)
ap = argparse.ArgumentParser()
#-------------------Start of EasyOCR--------------------#
ap.add_argument("-l", "--langs", type=str, default="en",
    help="comma separated list of languages to OCR")
ap.add_argument("-g", "--gpu", type=int, default=-1,
    help="whether or not GPU should be used")
args = vars(ap.parse_args())
# break the input languages into a comma separated list
langs = args["langs"].split(",")
print("[INFO] OCR'ing with the following languages: {}".format(langs))
def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()
def easyocr():
    if key & 0xFF==ord ('c'):
        cv2.imwrite("test.jpg", frame)
        image = cv2.imread("test.jpg")
        # OCR the input image using EasyOCR
        print("[INFO] OCR'ing input image...")
        reader = Reader(langs, gpu=args["gpu"] > 0)
        results = reader.readtext(image)
        # loop over the results
        for (bbox, text, prob) in results:
                # display the OCR'd text and associated probability
                print("[INFO] {:.4f}: {}".format(prob, text))
                # unpack the bounding box
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))
                # cleanup the text and draw the box surrounding the text along
                # with the OCR'd text itself
                text = cleanup_text(text)
                cv2.rectangle(image, tl, br, (0, 255, 0), 2)
                cv2.putText(image, text, (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                pyttsx3.speak(text)
        cv2.imshow("Image", image)
#------------------End of EasyOCR-----------------------#
#----------------Start of Money Initialize------------------#
lower1 = np.array([81,35,141])
upper1 = np.array([157,255,255])

lower2 = np.array([49,113,70])
upper2 = np.array([206,255,109])

Ratio = 0.90            # Rate of distance, greater will be checked easier


# With surf and sift we can use bf or flann, akaze only use akaze
#detector=cv2.xfeatures2d.SIFT_create()
#detector = cv2.xfeatures2d.SURF_create()
detectors = cv2.AKAZE_create()

#FLANN
FLANN_INDEX_KDITREE=0   #Procedures
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)   #Procedures
flann=cv2.FlannBasedMatcher(flannParam,{})  #Procedures

#BF
#BF = cv2.BFMatcher()

#AKAZE
AKAZE = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

# This is an array, each of the elements is a name directory of image.
# Dataset array
TraingIMGArr = ["TrainingData/100F.jpg","TrainingData/100B.jpg",
                "TrainingData/200F.jpg","TrainingData/200B.jpg"                ]

# Use to print to console and LCD
PrintingElement = ["100","100",
                    "200","200",
                    ]

print("WAITING TO GET FEATURE ...") #Print to console
#Loading features of dataset to DesArr 
DesArr = np.load("feature.npy",allow_pickle=True) 

print("START - PRESS BUTTON TO TAKE A PICTURE TO DETECT") #Print to console
#Print to LCD
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (420, 440)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
colors = (255, 0, 0)

# Line thickness of 2 px
thickness = 2
#----------------End of Money Initialize------------------#
#----------------Start of Face Initialize------------------#
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open("encoding.pickle", "rb").read())
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
##vs = VideoStream(src=0).start()
time.sleep(2.0)
# start the FPS counter
fps = FPS().start()
#----------------End of Face Initialize------------------#
def facerec():
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30))
    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    # loop over the facial embeddings
    for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                    encoding)
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1
                    # determine the recognized face with the largest number
                    # of votes (note: in the event of an unlikely tie Python
                    # will select first entry in the dictionary)
                    name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
            pyttsx3.speak(name)
            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                    (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
    # display the image to our screen
    #cv2.imshow("Frame", frame)
    # update the FPS counter
    fps.update()
    # stop the timer and display FPS information
def  Money():
    print("DETECTING ....... ")
    # Ready to take a picture
    #sleep(1) #delay 2sec let camera stable
    cv2.imwrite("userimg.jpg",frame)         
    # Read image has just taken
    Raw_usr_img=cv2.imread("userimg.jpg") #Read img from user (captured from raspberry)
    PhotoDetect = cv2.resize(Raw_usr_img, (640,480))
    PhotoDetect=PhotoDetect[130:420,40:630]
    hsv_img = cv2.cvtColor(PhotoDetect, cv2.COLOR_BGR2HSV)   # HSV image
    
    mask_sub1 = cv2.inRange(hsv_img , lower1, upper1)
    mask_sub2 = cv2.inRange(hsv_img , lower2, upper2)
    mask = cv2.bitwise_or(mask_sub1,mask_sub2)
    _,contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    suma=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        suma=suma+area
    result=suma/(PhotoDetect.shape[0]*PhotoDetect.shape[1])*100
    #print(result)
    if result>120:
        print ("PHOTO MONEY")
        #lcd_string(" PHOTO MONEY :) ",LCD_LINE_1,LCD_BACKLIGHT)
    else:
        Raw_usr_img=cv2.imread("userimg.jpg") #Read img from user (captured from raspberry)
        queryKP,queryDesc=detectors.detectAndCompute(Raw_usr_img,None) #Procedures to get feature from this picture
    
        max_point = 0; # Max point
        index_element_arr = 0; # Index which picture are detecting or detected to print out LCD or console

        # Print to LCD
        #lcd_string("   DETECTING:   ",LCD_LINE_1,LCD_BACKLIGHT)

        for i in range(len(TraingIMGArr)):            
            matches=AKAZE.knnMatch(queryDesc,DesArr[i],k=2) #Procedures 

            print("DETECTING - " + PrintingElement[i]) #Print to console which image are being processed
            #lcd_string("   "+PrintingElement[i] + " VND ",LCD_LINE_2,LCD_BACKLIGHT) #Print to LCD

            Match_Count = 0 # Create a variable to count match points from 2 images
            for m,n in matches:
                if(m.distance < Ratio * n.distance):   #If match 
                    Match_Count += 1    #increase by 1
            print(Match_Count)  #Print to console, comment if don't need it
            if Match_Count >= max_point: # If the Match_Count greater than max_point
                max_point = Match_Count  # Assign max_point again
                index_element_arr = i;   # Assign idex to print to console and LCD 
        
        #If box is empty, the match count usually < 30 MatchPoint
        if Match_Count > 24:
            #Print running time
            print("THAT IS - " + PrintingElement[index_element_arr]) #After run all dataset, print to console which money was detected
            #cv2.putText(frame,PrintingElement[index_element_arr],
            pyttsx3.speak(PrintingElement[index_element_arr])
            cv2.putText(frame, PrintingElement[index_element_arr], org, font, 
               fontScale, colors, thickness, cv2.LINE_AA)
        else:
            print("BOX IS EMPTY")
            cv2.putText(frame, 'EMPTY', org, font, 
               fontScale, colors, thickness, cv2.LINE_AA)
def color():
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([25,70,120])
    upper_yellow = np.array([30,255,255])

    lower_green = np.array([40,70,80])
    upper_green = np.array([10,255,255])

    lower_red = np.array([0,50,120])
    upper_red = np.array([10,255,255])

    lower_blue = np.array([90,60,0])
    upper_blue = np.array([121,255,255])

    mask1 = cv2.inRange(hsv, lower_yellow,upper_yellow)
    mask2 = cv2.inRange(hsv, lower_green, upper_green)
    mask3 = cv2.inRange(hsv, lower_red, upper_red)
    mask4 = cv2.inRange(hsv, lower_blue, upper_blue)

    cnts1 = cv2.findContours(mask1, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)

    cnts2 = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)

    cnts3 = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts3 = imutils.grab_contours(cnts3)

    cnts4 = cv2.findContours(mask4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts4 = imutils.grab_contours(cnts4)

    for c in cnts1:
        area1 = cv2.contourArea(c)
        if area1 > 5000:

            cv2.drawContours(frame,[c],-1,(0,255,0),3)

            M = cv2.moments(c)

            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            cv2.circle(frame,(cx,cy),7,(255,255,255),-1)
            cv2.putText(frame, "Yellow",(cx-20,cy-20),cv2.FONT_HERSHEY_SIMPLEX,2.5,(255,255,255),3)
            pyttsx3.speak("Yellow")

    for c in cnts2:
        area2 = cv2.contourArea(c)
        if area2 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Green", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
            pyttsx3.speak("Green")

    for c in cnts3:
        area3 = cv2.contourArea(c)
        if area3 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Red", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
            pyttsx3.speak("Red")

    for c in cnts4:
        area4 = cv2.contourArea(c)
        if area4 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Blue", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
            pyttsx3.speak("Blue")
while True:
    _,frame=camera.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    key=cv2.waitKey (1)
    if switch == 0:
        easyocr()
    elif switch ==1:
        color()
    elif switch ==2:
        Money()
    elif switch ==3:
        facerec()
    if key & 0xFF==ord ('e'):
        break
    if key & 0xFF==ord ('s'):
        cv2.destroyAllWindows()
        if switch < 3:
            switch+=1
            print (switch)
        else:
            switch =0
            print ("zERO",switch)
    cv2.imshow('img',frame)
camera.release()
cv2.destroyAllWindows()

