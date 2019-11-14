from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math

face_width_cm  = 13
face_height_cm = 12

point_sn = 33
point_ls = 51

"""
Converts a dimension from pixels to centimeters.
"""
def convert_px_to_cm(base_cm, base_px, target_px):
	return (base_cm / base_px) * target_px

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the face number
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	# Loops over the (x, y) coordinates and draw them on the image.
	for (x, y) in shape:
		cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

	print("+----------------------------------+")
	print("|       Relatório das medidas      |")
	print("+----------------------------------+")
	print("Sn: ", shape[point_sn])
	print("Ls: ", shape[point_ls])
	x1 = shape[point_sn][0]
	y1 = shape[point_sn][1]

	x2 = shape[point_ls][0]
	y2 = shape[point_ls][1]
	cv2.circle(image, (x1, y1), 3, (0, 255, 0), -1)
	cv2.circle(image, (x2, y2), 3, (0, 255, 0), -1)

	x = x2 - x1
	y = y1 - y2

	print(x, " ", y)
	distance = math.hypot(x, y)

	# Draws the line between two points (Sn & Ls).
	cv2.line(image, (x1, y1), (x2, y2), (0,255,0), 2) # Draw line

	distance_px = math.hypot(x, y)

	print("face_width_px: ", w)
	print("face_width_cm: ", face_width_cm)
	print("face_height_px:", h)
	print("face_height_cm:", face_height_cm)
	print("d(Sn, Ls) px: ", distance_px)
	print("d(Sn, Ls) cm: ", convert_px_to_cm(face_height_cm, h, distance_px))

	# print("Medida dos pontos Sn e Ls: ", distance)
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
