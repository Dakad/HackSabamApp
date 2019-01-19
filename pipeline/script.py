import argparse
import imutils
import cv2
import pytesseract

from improv import detect_contours, detect_edge, get_transform, deskew


def process_4_pt(**args):
    transform = None

    # Load the input image
    image = cv2.imread(args["image"])
    ratio = image.shape[0] / 500.0
    orig = image.copy()

    # Resize to speed up the processing
    img = imutils.resize(image, height=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("./process/4_pt_1img.jpg", img)
    cv2.imwrite("./process/4_pt_1gray.jpg", gray)

    contours = detect_contours(cv2.Canny(gray, 75, 200))
    if len(contours):
        print(" 4 pt Contour detected ")
        img2 = gray.copy()
        cv2.drawContours(img2, contours, -1, (0, 255, 0), 2)
        cv2.imwrite("./process/4_pt_2contours.jpg", img2)
        # cv2.imshow("Outline", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print(" Perspective Transform")
        (warped, effect) = get_transform(
            orig, contours[0], ratio, has_effect=True)

        cv2.imwrite("./process/4_pt_3warped.jpg", warped)

        warped_edges = cv2.Canny(warped, 75, 200)
        cv2.imwrite("./process/4_pt_3warped_edges.jpg", warped_edges)

        if effect is not None:
            cv2.imwrite("./process/4_pt_4effect.jpg", effect)

        transform = warped
    else:
        print(" No 4-pt contour found ")
        edged = detect_edge(img)

        print("Deskew ...")
        rotated = deskew(gray)['rotated']

        cv2.imwrite("./process/4_pt_2edged.jpg", edged)
        cv2.imwrite("./process/improv_deskew.jpg", rotated)

        transform = rotated

    config = ("-l eng --oem 1 --psm 6")
    resultat = pytesseract.image_to_string(transform, config=config)
    print(resultat)


def improv(**args):
    image = cv2.imread(args["image"])
    image = imutils.resize(image, height=750)

    # Grayscale the img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    correct = deskew(gray)

    # cv2.drawContours(tresh, coords, -1, (0, 255, 0), 2)
    #cv2.imshow("Original", image)
    cv2.imwrite("./process/improv_1img.jpg", image)

    #cv2.imshow("Dark", cv2.Canny(gray, 75, 200))
    cv2.imwrite("./process/improv_2gray.jpg", cv2.bitwise_not(gray))

    # cv2.putText(
    #     correct["rotated"],
    #     "Angle : {:.2f} c".format(correct["to"]),
    #     (10, 30),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.7,
    #     (255, 0, 0),
    #     2,
    # )
    # cv2.imshow("Rotated", correct["rotated"])
    print("Rotation Angle : {:.3f}°".format(correct["to"]))
    cv2.imwrite("./process/improv_3rotated.jpg", correct['rotated'])

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
    # improv(**args)
