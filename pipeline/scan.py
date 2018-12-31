from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


def detect_edge(image):
    # Convert the img to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur it
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Find the edges
    edged = cv2.Canny(gray, 75, 200)

    return edged


def detect_contours(image):
    # Find the contours in the image
    cnts = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # Keep the largest ones
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # ? Has 4 pts ?
        if len(approx) == 4:
            return [approx]
    # Otherwise, send
    return []


def get_four_point(image, contour, ratio):
    # Apply the 4-pt transform on the original image
    return four_point_transform(image, contour.reshape(4, 2) * ratio)


def transform(image, has_effect=False):
    # Convert warped img to GRAY
    warped = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if has_effect == True:
        # Threshold it
        T = threshold_local(warped, 11, offset=10, method="gaussian")
        # Apply 'black & white' paper effect
        effect = (warped > T).astype("uint8") * 255

    return (warped, effect)


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

print(" Evolution ")
edged = detect_edge(img)
## Show the original image, gray, blurried, edged detected imm
# cv2.imshow("Original", img)
# cv2.imshow("Edged", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

contours = detect_contours(edged.copy())
if len(contours) == 0:
    print(" No 4-pt contour found ")
    raise SystemExit(0)

print(" Contour ")
# cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
# cv2.imshow("Outline", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


print(" Perspective Transform")
four_point = get_four_point(orig, contours[0], ratio)
(warped, effect) = transform(four_point, has_effect=True)

cv2.imshow("Original", img)
cv2.imshow("Transform", imutils.resize(warped, height=500))
cv2.imshow("Transform & effect", imutils.resize(effect, height=500))
cv2.waitKey(0)
# cv2.destroyAllWindows()
