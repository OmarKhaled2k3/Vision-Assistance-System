import cv2
from easyocr import Reader
import argparse
import time
import numpy as np
import imutils
switch=2
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
    if c & 0xFF==ord ('c'):
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
detector = cv2.AKAZE_create()

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
        queryKP,queryDesc=detector.detectAndCompute(Raw_usr_img,None) #Procedures to get feature from this picture
    
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

    for c in cnts2:
        area2 = cv2.contourArea(c)
        if area2 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Green", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in cnts3:
        area3 = cv2.contourArea(c)
        if area3 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Red", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in cnts4:
        area4 = cv2.contourArea(c)
        if area4 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Blue", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
while True:
    _,frame=camera.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    c=cv2.waitKey (1)
    if switch == 0:
        easyocr()
    elif switch ==1:
        color()
    elif switch ==2:
        Money()
    if c & 0xFF==ord ('e'):
        break;
    cv2.imshow('img',frame)
camera.release()
cv2.destroyAllWindows()

