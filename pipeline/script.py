import argparse
import imutils
import cv2
import pytesseract

from four_point_transform import detect_contours, detect_edge, get_transform


def improv(**args):
    image = cv2.imread(args["image"])
    image = imutils.resize(image, height=750)

    # Grayscale the img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    correct = deskew(gray)

    cv2.putText(
        correct["rotated"],
        "Angle : {:.2f} c".format(correct["to"]),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    # cv2.drawContours(tresh, coords, -1, (0, 255, 0), 2)
    print("Rotation Angle : {:.3f}°".format(correct["to"]))
    cv2.imshow("Original", image)
    cv2.imshow("Dark", cv2.Canny(gray, 75, 200))
    cv2.imshow("Rotated", correct["rotated"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_4_pt(**args):
    # Load the input image
    image = cv2.imread(args["image"])
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    # Resize to speed up the processing
    img = imutils.resize(image, height=500)

    print(" Evolution ")
    edged = detect_edge(img)
    # Show the original image, gray, blurried, edged detected imm
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
    (warped, effect) = get_transform(
        orig, contours[0], ratio, has_effect=False)

    config = ("-l eng --oem 1 --psm 6")
    resultat = pytesseract.image_to_string(warped, config=config)
    print(resultat)

    cv2.imwrite("./process/transform_1img.jpg", img)

    # cv2.imshow("Original", img)
    cv2.imwrite("./process/transform_2edged.jpg", edged)
    # cv2.imshow("Transform", imutils.resize(warped, height=500))
    cv2.imwrite("./process/transform_3warped.jpg", warped)

    if effect != None:
        cv2.imwrite("./process/transform_4effect.jpg", effect)
        # cv2.imshow("Transform & effect", imutils.resize(effect, height=500))

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-i", "--image", required=True, help="Path to the image to be scanned"
    )
    args = vars(arg_parser.parse_args())
    process_4_pt(**args)
    improv(**args)
