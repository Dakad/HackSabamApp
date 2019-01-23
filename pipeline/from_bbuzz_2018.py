import numpy as np
import imutils
import argparse
import cv2
import matplotlib as plt

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("-i", "--image", required=True,
                       help="Path to input image file")
args = vars(arg_parse.parse_args())

image = cv2.imread(args["image"])
image = imutils.resize(image, height=750)

# Grayscale the img
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Flip the foreground
im = cv2.bitwise_not(gray)

cv2.imwrite("./process/0.jpg", image)

_, binary_thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV)

cv2.imwrite("./process/1.jpg", binary_thresh)


rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 100  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 600/2.0  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments
# line_image = np.copy(im) * 0  # creating a blank to draw lines on

lines = cv2.HoughLinesP(binary_thresh, rho, theta, threshold,
                        min_line_length, max_line_gap)

angle = 0


line_image = image.copy()

for line in lines:
    x1, y1, x2, y2 = line[0]
    r = np.arctan2(y2 - y1, x2 - x1)
    angle += np.arctan2(y2 - y1, x2 - x1)
    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

lines_edges = cv2.addWeighted(image.copy(), 0.8, line_image, 1, 0)
cv2.imwrite("./process/1b.jpg", lines_edges)

avg_radian = angle / len(lines)
avg_angle = avg_radian * 180 / np.pi

print("Average angle is %f °", avg_angle)

# Rotate the image
(height, width) = gray.shape[:2]
center = (width // 2, height // 2)
rot_mat = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
rotated = cv2.warpAffine(
    gray,
    rot_mat,
    (width, height),
    flags=cv2.INTER_LINEAR,
    # flags=cv2.INTER_CUBIC,
    borderMode=cv2.BORDER_REPLICATE,
)

cv2.imwrite("./process/2.jpg", rotated)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilate = cv2.morphologyEx(im, cv2.MORPH_DILATE, kernel)
cv2.imwrite("./process/3.jpg", dilate)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
connected = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel, iterations=2)
cv2.imwrite("./process/4.jpg", connected)
