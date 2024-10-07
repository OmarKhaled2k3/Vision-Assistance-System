import cv2
from easyocr import Reader
import argparse
import time
switch=0
camera = cv2.VideoCapture(0)
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--langs", type=str, default="en",
    help="comma separated list of languages to OCR")
ap.add_argument("-g", "--gpu", type=int, default=-1,
    help="whether or not GPU should be used")
args = vars(ap.parse_args())
# break the input languages into a comma separated list
langs = args["langs"].split(",")
print("[INFO] OCR'ing with the following languages: {}".format(langs))
def cam():
    
    _,img=camera.read()
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('img',img)
def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()
def easyocr():
    if c & 0xFF==ord ('q'):
        start = time.time()
        cv2.imwrite("test.jpg", img)
        image = cv2.imread("test.jpg")
        # OCR the input image using EasyOCR
        print("[INFO] OCR'ing input image...")
        reader = Reader(langs, gpu=args["gpu"] > 0)
        results = reader.readtext(image)
        # loop over the results
        end = time.time()
        print(end - start)
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
while True:
    _,img=camera.read()
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('img',img)
    c=cv2.waitKey (1)
    if switch == 0:
        easyocr()
