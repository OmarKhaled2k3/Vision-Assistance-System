import cv2
import numpy as np

def nothing(x):
    pass


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("HSV Value")
    cv2.createTrackbar("H MIN", "HSV Value", 0, 179, nothing)
    cv2.createTrackbar("S MIN", "HSV Value", 0, 255, nothing)
    cv2.createTrackbar("V MIN", "HSV Value", 0, 255, nothing)
    cv2.createTrackbar("H MAX", "HSV Value", 179, 179, nothing)
    cv2.createTrackbar("S MAX", "HSV Value", 255, 255, nothing)
    cv2.createTrackbar("V MAX", "HSV Value", 255, 255, nothing)

    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H MIN", "HSV Value")
        s_min = cv2.getTrackbarPos("S MIN", "HSV Value")
        v_min = cv2.getTrackbarPos("V MIN", "HSV Value")
        h_max = cv2.getTrackbarPos("H MAX", "HSV Value")
        s_max = cv2.getTrackbarPos("S MAX", "HSV Value")
        v_max = cv2.getTrackbarPos("V MAX", "HSV Value")

        lower_blue = np.array([h_min, s_min, v_min])
        upper_blue = np.array([h_max, s_max, v_max])

        hsv_min="MIN H:{} S:{} V:{}".format(h_min,s_min,v_min)
        hsv_max = "MAX H:{} S:{} V:{}".format(h_max, s_max, v_max)

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.putText(frame, hsv_min, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, hsv_max, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("HSV Value", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Frame Mask", result)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
