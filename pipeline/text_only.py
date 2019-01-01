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

# Create MSER object
mser = cv2.MSER_create()

vis = image.copy()

# detect regions in gray scale image
regions, _ = mser.detectRegions(gray)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(vis, hulls, 1, (0, 255, 0))

mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

for contour in hulls:
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

edged = cv2.Canny(gray, 75, 200)

# this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(image, edged, mask=mask)

cv2.imshow("Original", image)
cv2.imshow("text only", text_only)

cv2.waitKey(0)
cv2.destroyAllWindows()
