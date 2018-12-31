import numpy as np
import imutils
import argparse
import cv2


arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("-i", "--image", required=True, help="Path to input image file")
args = vars(arg_parse.parse_args())

image = cv2.imread(args["image"])
image = imutils.resize(image, height=750)

# Grayscale the img
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Flip the foreground
gray = cv2.bitwise_not(gray)
# Treshold, set foreground px to 255 and background to 0
tresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv2.imshow("Original", image)
cv2.imshow("Dark", tresh)
cv2.waitKey(0)
cv2.destroyWindow()
