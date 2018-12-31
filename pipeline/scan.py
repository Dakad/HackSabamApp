from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-i", "--image", required=True, help="Path to the image to be scanned"
)
args = vars(arg_parser.parse_args())

# Load the input image
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
# Resize to speed up the processing
img = imutils.resize(image, height=500)

# Convert the img to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur it
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# Find the edges
edged = cv2.Canny(gray, 75, 200)

# Show the original image, gray, blurried, edged detected imm
print(" Evolution ")
cv2.imshow("Original", img)
cv2.imshow("Gray", gray)
cv2.imshow("Blur", blur)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
