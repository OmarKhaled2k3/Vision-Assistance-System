from __future__ import print_function   #Use newest way to print if has new version in future
from __future__ import division         #Use newest way to division if has new version in future

from time import sleep  #import sleep lib as delay in Microcontroller
#import serial           #import Serial(UART) lib , need enable hardware uart in Rasp's setting

import numpy as np
import cv2              #import opencv lib
#import smbus
import time

cap = cv2.VideoCapture(1)
cap.set(3, 320) # set video width
cap.set(4, 240) # set video height
###############DETECT CODE###########################################################################

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
                "TrainingData/200F.jpg","TrainingData/200B.jpg",
                "TrainingData/20F.jpg","TrainingData/20B.jpg",
                "TrainingData/50F.jpg","TrainingData/50B.jpg",
                "TrainingData/5F.jpg","TrainingData/5B.jpg",
                ]# Use to print to console and LCD
PrintingElement = ["100","100",
                    "200","200",
                   "20","20",
                   "50","50",
                   "5","5"
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
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2
while(1):
    ret, frame = cap.read()
    key = cv2.waitKey(1)
    if key != 27: # If press button
        while key == 27: # While press button (don't do anything)
            {}
        # Get start time
        start = time.time()
        # Print to LCD 
        #lcd_string("",LCD_LINE_2,LCD_BACKLIGHT)
        #lcd_string(" GETTING IMAGE! ",LCD_LINE_1,LCD_BACKLIGHT)

        # Turn on WHITE LED inside box (set LOW because i'm using LOW trigger for turn ON led)
        #GPIO.output(24, GPIO.LOW)

        # Print to LCD 
        print("DETECTING ....... ")
        
        # Ready to take a picture
        ret, frame = cap.read()
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
        #cv2.imwrite("m1.jpg",mask_sub1)
        #cv2.imwrite("m2.jpg",mask_sub2)
        #cv2.imwrite("hsv.jpg",hsv_img)
        #cv2.imwrite("m.jpg",mask)
        _,contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        suma=0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            suma=suma+area
        result=suma/(PhotoDetect.shape[0]*PhotoDetect.shape[1])*100
        #print(result)
        if result>120:
            print ("PHOTO MONEY")
            cv2.putText(frame, 'PHOTO MONEY', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
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
            # Get end time
            end = time.time()
            print(end - start)
            
            #If box is empty, the match count usually < 30 MatchPoint
            if Match_Count > 24:
                #Print running time
                print("THAT IS - " + PrintingElement[index_element_arr]) #After run all dataset, print to console which money was detected
                #cv2.putText(frame,PrintingElement[index_element_arr],
                cv2.putText(frame, PrintingElement[index_element_arr], org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
                #Print to LCD
                #lcd_string("    THAT IS:    ",LCD_LINE_1,LCD_BACKLIGHT)
                #lcd_string("   "+PrintingElement[index_element_arr] + " VND ",LCD_LINE_2,LCD_BACKLIGHT)
            else:
                print("BOX IS EMPTY")
                cv2.putText(frame, 'EMPTY', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
                #lcd_string("  BOX IS EMPTY  ",LCD_LINE_1,LCD_BACKLIGHT)
                #lcd_string(" ",LCD_LINE_2,LCD_BACKLIGHT)
        cv2.imshow("Frame",frame)        
            
