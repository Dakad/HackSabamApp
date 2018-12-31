import argparse


def detect_edge(image):
    return NotImplementedError


def detect_contours(image):
    raise NotImplementedError


def get_transform(image, contour, ratio, has_effect=False):
    raise NotImplementedError


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
(warped, effect) = get_transform(orig, contours[0], ratio, has_effect=True)


cv2.imshow("Original", img)
cv2.imshow("Transform", imutils.resize(warped, height=500))
cv2.imshow("Transform & effect", imutils.resize(effect, height=500))
cv2.waitKey(0)
# cv2.destroyAllWindows()
