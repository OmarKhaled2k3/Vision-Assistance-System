import cv2
import argparse
import time
import numpy as np
import imutils
import pickle
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import pyttsx3
from picamera.array import PiRGBArray
from picamera import PiCamera
#engine = pyttsx3.init() # object creation
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
switch=0
ap = argparse.ArgumentParser()

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
def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue
			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)
def ocrfunc():
    if key & 0xFF==ord ('c'):
        cv2.imwrite("test.jpg", frame)
        image = cv2.imread("test.jpg")
        orig = image.copy()
        (origH, origW) = image.shape[:2]
        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (args["width"], args["height"])
        rW = origW / float(newW)
        rH = origH / float(newH)
        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]
        # define the two output layer names for the EAST detector model that
        # we are interested in -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
                "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"]
        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")
        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        # initialize the list of results
        results = []
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            # in order to obtain a better OCR of the text we can potentially
            # apply a bit of padding surrounding the bounding box -- here we
            # are computing the deltas in both the x and y directions
            dX = int((endX - startX) * args["padding"])
            dY = int((endY - startY) * args["padding"])
            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(origW, endX + (dX * 2))
            endY = min(origH, endY + (dY * 2))
            # extract the actual padded ROI
            roi = orig[startY:endY, startX:endX]
            # in order to apply Tesseract v4 to OCR text we must supply
            # (1) a language, (2) an OEM flag of 1, indicating that the we
            # wish to use the LSTM neural net model for OCR, and finally
            # (3) an OEM value, in this case, 7 which implies that we are
            # treating the ROI as a single line of text
            config = ("-l eng --oem 1 --psm 7")
            text = pytesseract.image_to_string(roi, config=config)
            # add the bounding box coordinates and OCR'd text to the list
            # of results
            results.append(((startX, startY, endX, endY), text))
        # sort the results bounding box coordinates from top to bottom
        results = sorted(results, key=lambda r:r[0][1])
        # loop over the results
        end = time.time()
        print(end - start)
        for ((startX, startY, endX, endY), text) in results:
            # display the text OCR'd by Tesseract
            print("OCR TEXT")
            print("========")
            print("{}\n".format(text))
            # strip out non-ASCII text so we can draw the text on the image
            # using OpenCV, then draw the text and a bounding box surrounding
            # the text region of the input image
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            output = orig.copy()
            cv2.rectangle(output, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
            cv2.putText(output, text, (startX, startY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            # show the output image
            cv2.imshow("Text Detection", output)
            pyttsx3.speak(text)
    

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
for fram in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = fram.array
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    key=cv2.waitKey (1)
    if switch == 0:
        color()
    elif switch ==1:
        Money()
    elif switch ==2:
        facerec()
    elif switch ==3:
        ocrfunc()
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
    rawCapture.truncate(0)
cv2.destroyAllWindows()

