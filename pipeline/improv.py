from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils


# https://github.com/yardstick17/image_text_reader/blob/master/image_preprocessing/remove_noise.py

def _order_points_(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = _order_points_(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def detect_edge(image):
    # Convert the img to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur it
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
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


def get_transform(image, contour, ratio, has_effect=False):
    # Apply the 4-pt transform on the original image
    four_point = four_point_transform(image, contour.reshape(4, 2) * ratio)
    # Convert warped img to GRAY
    warped = cv2.cvtColor(four_point, cv2.COLOR_BGR2GRAY)

    effect = None
    if has_effect == True:
        # Threshold it
        T = threshold_local(warped, 11, offset=10, method="gaussian")
        # Apply 'black & white' paper effect
        effect = (warped > T).astype("uint8") * 255

    return (warped, effect)


def deskew(image, gray):
    """A Skewed image is defined as a image which is not straight.
    Skewed images directly impact the line segmentation of OCR engine which reduces its accuracy

    """

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

    rotated = None

    if(np.abs(angle) != 0):
        # Rotate the image
        (height, width) = gray.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return [rotated, angle]


def remove_shadow(image):
    result_planes = []
    rgb_planes = cv2.split(image)

    for plan in rgb_planes:
        img_dilated = cv2.dilate(image, np.ones((7, 7), np.uint8))
        img_bg = cv2.medianBlur(img_dilated, 21)
        img_diff = 255 - cv2.absdiff(plane, img_bg)
        img_norm = cv2.normalize(
            img_diff, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(img_diff)
    result = cv2.merge(result_planes)
    return result


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise(image):
    """ Noise is a random variation of color or brightness btw pixels.
        Noise decrease the readability of text from an image. 
        There are two major types of noises : 
            Salt & Pepper
            Gaussian 
    """
    filtered = cv2.absdiff(image.astype(np.uint8), 255,
                           cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    img = image_smoothening(image)
    transform = cv2.bitwise_or(img, closing)
    return transform


def kmeans(input_img, k, i_val):
    # _, thresh = cv2.threshold(img, kmeans(input_img=img, k=8, i_val=2)[0], 255, cv2.THRESH_BINARY)

    hist = cv2.calcHist([input_img], [0], None, [256], [0, 256])
    img = input_img.ravel()
    img = np.reshape(img, (-1, 1))
    img = img.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(
        img, k, None, criteria, 10, flags)
    centers = np.sort(centers, axis=0)

    return centers[i_val].astype(int), centers, hist
