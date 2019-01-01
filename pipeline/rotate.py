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

# Get the (x,y) coord
coords = np.column_stack(np.where(tresh > 0))

# Get the rotated bouding box of these coords
# https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
angle = cv2.minAreaRect(coords)[-1]  # Return val btwn [-90,0]

# As the rect rotate clockwise, angle --> 0
if angle < -45:
    angle = -(90 + angle)
else:
    # Just take the inverse
    angle = -angle

# Rotate the image
(height, width) = image.shape[:2]
center = (width // 2, height // 2)
matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(
    image,
    matrix,
    (width, height),
    flags=cv2.INTER_CUBIC,
    borderMode=cv2.BORDER_REPLICATE,
)

cv2.putText(
    rotated,
    "Angle : {:.2f} c".format(angle),
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 0, 0),
    2,
)

# cv2.drawContours(tresh, coords, -1, (0, 255, 0), 2)
print("Rotation Angle : {:.3f}°".format(angle))
cv2.imshow("Original", image)
cv2.imshow("Dark", cv2.Canny(gray, 75, 200))
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
