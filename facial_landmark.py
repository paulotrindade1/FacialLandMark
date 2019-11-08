from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math

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

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image

	ponto1 = int(input("Informe o ponto 1: "))
	ponto2 = int(input("Informe o ponto 2: "))
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 0), -1)


	print("Sn: ", shape[ponto1])
	print("Ls: ", shape[ponto2])
	x1 = shape[ponto1][0]
	y1 = shape[ponto1][1]

	x2 = shape[ponto2][0]
	y2 = shape[ponto2][1]
	cv2.circle(image, (x1, y1), 1, (0, 255, 0), -1)
	cv2.circle(image, (x2, y2), 1, (0, 255, 0), -1)

	x = x2 - x1
	y = y1 - y2

	print(x, " ", y)
	distance = math.hypot(x, y)
	
	cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
	cv2.putText(image, "test", (x1 + 5, y1 + 5),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
	print("Distancia dos pontos Sn e Ls: ", distance)
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
