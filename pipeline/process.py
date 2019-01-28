import argparse
import imutils
import cv2
import math
import pytesseract

from config import Config
from .improv import detect_contours, detect_edge, get_transform, deskew


class Optimiser(object):
    """To Handle the optimisation process of the uploaded image

    Arguments:


    Returns:
        Unknown -- The optimised image for the OCR
    """

    def __init__(self, **args):
        self._img_path = args['image']
        self._img = cv2.imread(args["image"])
        self._img_original = self.img.copy()
        self._mser = cv2.MSER_create()

    def _process_4_pt(self):
        transform = None

        # Resize to speed up the processing
        self._img = imutils.resize(self._img, height=Config.IMG_RESIZE_HEIGHT)
        self._ratio = self._img.shape[0] / Config.IMG_RESIZE_HEIGHT
        self._gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

        if Config.DEBUG:
            cv2.imwrite("./process/4_pt_1img.jpg", self._img)
            cv2.imwrite("./process/4_pt_1gray.jpg", self._gray)

        contours = detect_contours(cv2.Canny(self._gray, 75, 200))

        has_contours = bool(len(contours))
        if has_contours:
            img2 = self._gray.copy()
            cv2.drawContours(img2, contours, -1, (0, 255, 0), 2)
            if Config.DEBUG:
                cv2.imwrite("./process/4_pt_2contours.jpg", img2)

            (warped, effect) = get_transform(
                self._img_original, contours[0], self._ratio, has_effect=True)

            if Config.DEBUG:
                cv2.imwrite("./process/4_pt_3warped.jpg", warped)

            warped_edges = cv2.Canny(warped, 75, 200)

            if Config.DEBUG:
                cv2.imwrite("./process/4_pt_3warped_edges.jpg", warped_edges)

            if Config.DEBUG and effect is not None:
                cv2.imwrite("./process/4_pt_4effect.jpg", effect)

            transform = warped

        return [has_contours, transform]

    def _apply_deskew(self):
        (rotated, angle) = deskew(self._img, self._gray)

        if(math.fabs(angle) == 0):
            return
        if Config.DEBUG:
            print("Rotation Angle : {:.3f}°".format(angle))
            cv2.imwrite("./process/improv_2rotated.jpg", rotated)

        self._img = rotated

    def _detect_text_regions(self):
        inverse = cv2.bitwise_not(self._gray)

        # Detect regions in gray scale image
        regions, _ = mser.detectRegions(inverse)

        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        if Config.DEBUG:
            vis = self._img.copy()
            cv2.polylines(vis, hulls, 1, (0, 255, 0))
            cv2.imwrite("./process/1txt_lines.jpg", vis)

        mask = np.zeros((image.shape[0], self.img.shape[1], 1), dtype=np.uint8)

        for contour in hulls:
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

        edged = cv2.Canny(inverse, 75, 200)

        # this is used to find only text regions, remaining are ignored
        text_only = cv2.bitwise_and(inverse, edged, mask=mask)
        if Config.DEBUG:
            cv2.imwrite("./process/2txt_only.jpg", text_only)

        return text_only

    def exec(self):

        (has_4_pth, transform) = self._process_4_pt()
        if not has_4_pth:
            if Config.DEBUG:
                print(" 4 pt Contour not detected ")
                # Resize to speed up the processing
                self._img = imutils.resize(
                    self._img, height=Config.IMG_RESIZE_HEIGHT)
                self._gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
                self._ratio = self._img.shape[0] / Config.IMG_RESIZE_HEIGHT

        self._apply_deskew()

        self.transform = self._detect_text_regions()

        return transform


def ocr(transform, lang='eng', psm=6) -> str:
    # OCR options :
    # --oem NUM     Specify OCR Engine mode.
    # OCR Engine modes:
    #   0    Legacy engine only.
    #   1    Neural nets LSTM engine only.
    #   2    Legacy + LSTM engines.

    # --psm NUM     Specify page segmentation mode.
    # Page segmentation modes:
    #  4    Assume a single column of text of variable sizes.
    #  5    Assume a single uniform block of vertically aligned text.
    #  6    Assume a single uniform block of text.
    #  7    Treat the image as a single text line.

    config = ("-l %s --oem 1 --psm %d" % (lang, psm))
    resultat = pytesseract.image_to_string(transform, config=config)
    return resultat


def parse(text: str) -> str:
    # Strip out non-ASCII text
    parsed = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    return parsed


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-i", "--image", required=True, help="Path to the image to be scanned"
    )
    args = vars(arg_parser.parse_args())

    transform = Optimiser(**args).exec()

    text = ocr(transform)
    print("Result : \n", text)
