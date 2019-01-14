import cv2
import numpy as np
from math import sqrt


def thresh(in_im):
	# Thresh using Adaptive Thresholding
	cv2.GaussianBlur(in_im,(5,5),1)
	gray = cv2.cvtColor(in_im,cv2.COLOR_BGR2GRAY)
	thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	cv2.GaussianBlur(thresh,(5,5),1)
	return thresh

def edge_detect(in_im):
	# Thresh using Edge Detection
	cv2.GaussianBlur(in_im,(5,5),1)
	gray = cv2.cvtColor(in_im,cv2.COLOR_BGR2GRAY)
	canny = cv2.Canny(in_im,100,200)
	cv2.GaussianBlur(canny,(5,5),1)
	return canny


def resize(in_im):
	# Resize image to appt size
	row,col = in_im.shape[:-1]
	if row > 2000 or col > 2000:
		in_im = cv2.resize(in_im,(int(col/2),int(row/2)),interpolation = cv2.INTER_CUBIC)

	return in_im

def deletepoints(approx,thresh_value):
	# Delete redundant points while detection table from image
	idx= 0
	n = len(approx)
	while idx < n:
		idx2 = 0
		while idx2 < n:
			if approx[idx] == approx[idx2]:
				idx2 += 1
				continue

			if distance(approx[idx][0],approx[idx2][0]) < thresh_value :
				# print ("Deleted")
				del approx[idx2]
				n -= 1
				continue

			idx2 += 1
		idx += 1
	return approx



def distance(pt1,pt2):	# Find distance between 2 points
	return sqrt((pt1[0]-pt2[0])**2 + (pt1[1] - pt2[1])**2 )


def draw(in_im):
	row,col = in_im.shape[:-1]
	copy = in_im.copy()
	canny = edge_detect(in_im)
	canny=cv2.dilate(canny, np.ones((7, 7), np.uint8), iterations=1)

	# Find contours on Image
	_, c, h = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cMax = max(c, key = cv2.contourArea)
	# print(cMax)

	# Polygon Approximation to find corners of table
	epsilon = 0.05*cv2.arcLength(cMax,True)
	approx = cv2.approxPolyDP(cMax,epsilon,True).tolist()
	approx.sort(key = lambda x : sqrt(x[0][0]**2 + x[0][1]**2))
	# print("Points detected : " , len(approx))
	#deletepoints(approx,int(min(row,col)/10))
	# print("After deletion, : ", len(approx))

	x,y,w,h = cv2.boundingRect(cMax)		# Find the Bounding Rectangle
	# cv2.rectangle(in_im,(x,y),(x+w,y+h),(0,0,255),2)


	for i in approx :
		cv2.circle(in_im,(i[0][0],i[0][1]),5,(0,255,0),-1)		# Draw Table corners

	if len(approx)==4:
		print ("Table Detected")

		pts1 = np.float32([i[0] for i in approx])
		# print (pts1)
		if w>=h:		# Horizontal Table
			print ("Horizontal")
			pts2 = np.float32([[0,0],[0,h],[w,0],[w,h]])
			M = cv2.getPerspectiveTransform(pts1,pts2)
			roi = cv2.warpPerspective(in_im,M,(w,h))
		else:
			print ("Vertical")		# Vertical Table
			pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
			M = cv2.getPerspectiveTransform(pts1,pts2)
			roi = cv2.warpPerspective(in_im,M,(w,h))

	else:
		print("Returning original Image, Points : ",len(approx))
		cv2.drawContours(in_im, cMax, -1, (255, 0, 0), 1)
		roi = copy

	#cv2.imshow("Input",in_im)
	# cv2.waitKey(0)
	return roi
