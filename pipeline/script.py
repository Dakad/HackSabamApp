import argparse
import imutils
import cv2
import math
import pytesseract

from improv import detect_contours, detect_edge, get_transform, deskew


def process_4_pt(**args):
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


def apply_deskew(transform, gray):
    (rotated, angle) = deskew(transform, gray)

    if(math.fabs(angle) == 0):
        return transform
    print("Rotation Angle : {:.3f}°".format(angle))
    cv2.imwrite("./process/improv_2rotated.jpg", rotated)
    return rotated


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


def main(**args):
    opts = dict()

    opts['image'] = args['image']
    opts['img'] = cv2.imread(args["image"])
    opts['img_original'] = opts['img'].copy()

    (has_4_pth, transform) = process_4_pt(**opts)
    if has_4_pth:
        print(" 4 pt Contour detected ")
        opts['img'] = transform
    else:
        # Resize to speed up the processing
        opts['img'] = imutils.resize(opts['img'], height=750)

    opts['gray'] = cv2.cvtColor(opts['img'], cv2.COLOR_BGR2GRAY)
    opts['ratio'] = opts['img'].shape[0] / 750.0

    transform = apply_deskew(opts['img'], opts['gray'])

    text = ocr(transform)
    print("Result : \n", text)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-i", "--image", required=True, help="Path to the image to be scanned"
    )
    args = vars(arg_parser.parse_args())
    main(**args)
