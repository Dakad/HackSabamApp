import argparse


def main(**args):
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


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument(
        "-i", "--image", required=True, help="Path to input image file"
    )
    parsed = vars(arg_parse.parse_args())
    main(parsed)
