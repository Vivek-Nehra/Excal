import cv2
import numpy as np
from math import sqrt


def edge_detect(img):
	cv2.GaussianBlur(img,(5,5),1)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	canny = cv2.Canny(img,100,200)
	cv2.GaussianBlur(canny,(5,5),1)
	# print(canny.shape,canny.dtype)
	return canny


def draw(in_im):
	copy = in_im.copy()
	canny = edge_detect(in_im)
	cv2.GaussianBlur(canny,(5,5),True)
	cv2.medianBlur(canny,3)
	canny=cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1)

	_, c, h = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cMax = max(c, key = cv2.contourArea)
	# print(cMax)

	epsilon = 0.05*cv2.arcLength(cMax,True)
	approx = cv2.approxPolyDP(cMax,epsilon,True).tolist()
	approx.sort(key = lambda x : sqrt(x[0][0]**2 + x[0][1]**2))

	x,y,w,h = cv2.boundingRect(cMax)
	cv2.rectangle(in_im,(x,y),(x+w,y+h),(0,0,255),2)

	cv2.drawContours(in_im, cMax, -1, (255, 0, 0), 1)
	print(approx)

	for i in approx :
		cv2.circle(in_im,(i[0][0],i[0][1]),5,(0,255,0),-1)

	if len(approx)==4:
		print ("Table Detected")

		pts1 = np.float32([[i[0] for i in approx]])
		pts2 = np.float32([[0,0],[0,h],[w,0],[w,h]])
		M = cv2.getPerspectiveTransform(pts1,pts2)
		roi = cv2.warpPerspective(in_im,M,(w,h))

	else:
		print("Returning original Image")
		roi = copy
	#
	# cv2.imshow("Input",in_im)
	# cv2.imshow("Canny",canny)
	cv2.waitKey(0)
	return roi

# extLeft = tuple(cMax[cMax[:, :, 0].argmin()][0])
# extRight = tuple(cMax[cMax[:, :, 0].argmax()][0])
# extTop = tuple(cMax[cMax[:, :, 1].argmin()][0])
# extBot = tuple(cMax[cMax[:, :, 1].argmax()][0])
# cv2.circle(in_im, extLeft, 8, (0, 0, 255), -1)
# cv2.circle(in_im, extRight, 8, (0, 0, 255), -1)
# cv2.circle(in_im, extTop, 8, (0, 0, 255), -1)
# cv2.circle(in_im, extBot, 8, (0, 0, 255), -1)
# print(extLeft,extRight,extTop,extBot)
# cmaxx =0
# cmaxy = 0
# cminx = 10000
# cminy = 10000
# for cnt in cMax:
# 	# print(cnt[0])
# 	cx,cy = cnt[0]
# 	cminx = min(cminx,cx)
# 	cmaxx = max(cmaxx,cx)
# 	cminy = min(cminy,cy)
# 	cmaxy = max(cmaxy,cy)
# print(cminx,cminy)
# print(cmaxx,cmaxy)

# cv2.line(in_im,(cminx,cminy),(cminx,cmaxy),(255,0,0),5)
# cv2.drawContours(in_im, [cMax], cMax[-1], (255, 0, 0), 3)
# print(cMax[0],cMax[-1])
# cx1,cy1 = cMax[0][0]
# cx2,cy2 = cMax[-1][0]
# print(cx2, cy2)
# print(cMax)
# print(x,y)
# for cnt in c:
# 	M = cv2.moments(cnt)
# 	if(M['m00']>800) and :
# 		cx = int(M['m10']/M['m00'])
# 		cy = int(M['m01']/M['m00'])
# 		cv2.drawContours(in_im, c, -1, (255, 0, 0), 2)


# cv2.imshow("Original",in_im)
# cv2.imshow("Binary",canny)
# # cv2.imshow("Eroded",eroded)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
