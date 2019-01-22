import argparse
import imutils
import cv2
import math
import pytesseract

from improv import detect_contours, detect_edge, get_transform, deskew


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

    def _process_4_pt(**args):
        transform = None

        # Resize to speed up the processing
        img = imutils.resize(args['img'], height=750)
        ratio = args['img'].shape[0] / 500.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite("./process/4_pt_1img.jpg", img)
        cv2.imwrite("./process/4_pt_1gray.jpg", gray)

        contours = detect_contours(cv2.Canny(gray, 75, 200))

        has_contours = len(contours) > 1
        if has_contours:
            img2 = gray.copy()
            cv2.drawContours(img2, contours, -1, (0, 255, 0), 2)
            cv2.imwrite("./process/4_pt_2contours.jpg", img2)

            (warped, effect) = get_transform(
                args["img_original"], contours[0], ratio, has_effect=True)

            cv2.imwrite("./process/4_pt_3warped.jpg", warped)

            warped_edges = cv2.Canny(warped, 75, 200)
            cv2.imwrite("./process/4_pt_3warped_edges.jpg", warped_edges)

            if effect is not None:
                cv2.imwrite("./process/4_pt_4effect.jpg", effect)

            transform = warped
        else:
            print(" No 4-pt contour found ")

        return [has_contours, transform]

    def _apply_deskew(transform, gray):
        (rotated, angle) = deskew(transform, gray)

        if(math.fabs(angle) == 0):
            return transform
        print("Rotation Angle : {:.3f}°".format(angle))
        cv2.imwrite("./process/improv_2rotated.jpg", rotated)
        return rotated

    def exec(self):

        (has_4_pth, transform) = self._process_4_pt(**opts)
        if has_4_pth:
            print(" 4 pt Contour detected ")
            self._img = transform
        else:
            # Resize to speed up the processing
            self._img = imutils.resize(self._img, height=750)

        self._gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        ratio = self._img.shape[0] / 750.0

        transform = self._apply_deskew(self._img, self._gray)

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
