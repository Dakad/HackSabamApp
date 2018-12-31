#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2


def decode_predictions(scores, geometry):
    """ Detect regions of text

    Parameters
    ----------
    scores 
        Probabilities for positive text regions
    geometry
        The bounding boxes of the text regions
    
    Returns
    -------
    tuple
        (bounding box locations of the text, corresponding probability of that region containing text)
    """

    # Grab the number of rows and columns from the scores volume,
    # Init our set of bounding box rectangles and matching
    # confidence scores
    (num_rows, num_cols) = scores.shape[2:4]
    rects = []
    confidences = []

    # Iterate over the number of rows
    for y in range(0, num_rows):
        # Extract the scores (proba)
        # the geometrical data to derive potential bounding box
        # coordinates surronding the text
        score_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]

        # Iterate over the number of rows
        for x in range(0, num_cols):
            # Ignore score with sufficient proba
            if score_data[x] < args["min_confidence"]:
                continue

            # Compute the offset factor as resulting feaeture
            # maps will be 4x smaller than the input range
            (x_offset, y_offset) = (x * 4.0, y * 4.0)

            # Extract the rotation ange for the prediction
            # Compute the sinus & consinus
            angle = angles[x]
            cosinus = np.cos(angle)
            sinus = np.sin(angle)

            # Use the geometry volume to derive the bounding box 'width and height
            height = x_data0[x] + x_data2[x]
            width = x_data1[x] + x_data3[x]

            # Compute both the starting and engin (x,y)-coordinates
            # for the text prediction bounding box
            x_end = int(x_offset + (cosinus * x_data1[x]) + (sinus * x_data2[x]))
            y_end = int(y_offset + (cosinus * x_data1[x]) + (sinus * x_data2[x]))
            x_start = int(x_end - width)
            y_start = int(x_end - height)

            # Add the bouding box coordinates and proba score to list
            rects.append((x_start, y_start, x_end, y_end))
            confidences.append(score_data[x])

    # Returnn (bouding box, confidences)
    return (rects, confidences)


# construct the argument parser and parse the arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-i", "--image", type=str, help="Path to input image")
# --east : The path to the pre-trained EAST text detector
arg_parser.add_argument(
    "-e", "--east", type=str, help="path to input EAST text detector"
)
# --min-confidence : The minimum probability of a detected text region
arg_parser.add_argument(
    "-c",
    "--min-confidence",
    type=float,
    default=0.5,
    help="Minimum probability required to inspect a region",
)
arg_parser.add_argument(
    "-wd",
    "--width",
    type=int,
    default=320,
    help="Nearest multiple of 32 for resized width",
)
arg_parser.add_argument(
    "-ht",
    "--height",
    type=int,
    default=320,
    help="Nearest multiple of 32 for resized height",
)
arg_parser.add_argument(
    "-p",
    "--padding",
    type=float,
    default=0.0,
    help="Amount of padding to add to each border of ROI",
)

args = vars(arg_parser.parse_args())

# Load the input image
image = cv2.imread(args["image"])
orig = image.copy()
# Get the dimensions
(orig_height, orig_width) = image.shape[:2]
 
# Set the new width & height
(new_width, new_height) = (args["width"], args["height"])

# Calculate ratios to use to scale our bounding box coordinates
ratio_width = orig_width / float(new_width)
ratio_height = orig_height / float(new_height)
 
# Reesize the image ignoring aspect-ratio
img = cv2.resize(image, (new_width, new_height))
(img_height, img_width) = img.shape[:2]


# Define the two output layer names for the EAST detector model
layers = [
	"feature_fusion/Conv_7/Sigmoid", # Output probabilities 
	"feature_fusion/concat_3" # Derive the bounding box coordinates of text
]

if(args["east"]):
    # Load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(args["east"])


# Construct blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (img_width, img_height),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layers)

(rects, confidences) = decode_predictions(scores, geometry)

# Non-maxima suppression to suppress weak, overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

results = []
for (x_start, y_start, x_end, y_end) in boxes:
    # Scale the bouding box coordinates based on the respective ratios
    x_start = int(x_start * ratio_width)
    y_start = int(y_start * ratio_height)
    x_end = int(x_end * ratio_width)
    y_end = int(y_end * ratio_height)

    # To impovre the OCR, apply a bit padding surronding the bounding box
    x_delta = int((x_end - x_start) * args["padding"])
    y_delta = int((y_end - y_start) * args["padding"])

    # Apply padding to each side of the bounding box
    x_start = max(0, x_start - x_delta)
    y_start = max(0, y_start - y_delta)
    x_end = min(orig_width, x_end + (x_delta * 2))
    y_end = min(orig_height, y_end + (y_delta * 2))

    # Extract the actual padded ROI
    roi = orig[y_start:y_end, x_start:x_end]

    # To apply Tesseract v4 to OCR text, provide
	# (1) a language, 
    # (2) an OEM flag of 4, to use the LSTM Neural Net Model for OCR
	# (3) an OEM value, in this case, 7 to treat the ROI as a single line of text
    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(roi, config=config)
 
	# add the bounding box coordinates and OCR'd text to the list
	# of results
    results.append(((x_start, y_start, x_end, y_end), text))


# Sort the results bounding box coordinates from top to bottom
results = sorted(results, key=lambda r:r[0][1])

# Loop over the results
for (x_start, y_start, x_end, y_end) in results:
    print("OCR TEXT : \n")
    print("{}\n".format(text))

    # Strip out non-ASCII text so we can draw the text on the image
	# Using OpenCV, then draw the text and a bounding box surrounding
	# the text region of the input image
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    output = orig.copy()
    cv2.rectangle(output, (startX, startY), (endX, endY),(0, 0, 255), 2)
    cv2.putText(output, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3) 
	
    # Show the output image
    cv2.imshow("Text Detection", output)
    cv2.waitKey(0)