#!/usr/bin/env python
import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.show()
def main():
	cap = cv2.VideoCapture(0)
	while(cap.isOpened()):
		ret, img = cap.read()
		blur = cv2.GaussianBlur(img, (3, 3), 0)
		skinMask = HSVBin(blur)
		contours = getContours(skinMask)
		if(contours):
			center, radius = getSkinContourCircle(contours)
			cv2.circle(img,center,radius,(0,255,0),2)
			# cimg = cv2.cvtColor(blur,cv2.COLOR_GRAY2BGR)
			circles, crop = getCircles(blur, contours)
			cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
			cv2.imshow('capture', np.hstack([img, circles]))
			height, width, channels = crop.shape
			if(height > 0 and width > 0):
				hist = cv2.calcHist([crop],[0],None,[256],[0,256])
				plt.plot(hist)				
				cv2.imshow('croped', np.hstack([crop]))
			k = cv2.waitKey(10)
			if k == 27:
				break
			if k == 83 or k == 115:
				print(k)
				plt.show()
				print(hist)
				np.savetxt('hist.txt', hist)
		else:
			print('No skin to detect')


def getContours(img):
	kernel = np.ones((5, 5), np.uint8)
	closed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
	img, contours, h = cv2.findContours(
		closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	validContours = []
	for cont in contours:
		if cv2.contourArea(cont) > 9000:
			validContours.append(cv2.convexHull(cont))
	return validContours

def getHistogram(img):
	hist = cv2.calcHist([img], [0], None, [256], [0,256])
	return hist

def getSkinContourCircle(contours):
	circle = max(contours, key=cv2.contourArea)
	(x,y),radius = cv2.minEnclosingCircle(circle)
	center = (int(x),int(y))
	radius = int(radius)
	return [center, radius]


def getCircles(blur, contours):
	output = blur.copy()
	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
	
	ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
	heigth, width, = gray.shape
	mask = np.zeros((heigth, width), np.uint8)
	
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
	
	if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
 
		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			cv2.circle(output, (x, y), r, (0, 255, 0), 4)
			cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
			# Mask
			r=r+4
			# Draw on mask
			cv2.circle(mask,(x,y),r,(255,255,255),thickness=-1)
	masked_data = cv2.bitwise_and(blur, blur, mask=mask)
	# Apply Threshold
	_,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

	# Find Contour
	contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	x,y,w,h = cv2.boundingRect(contours[0])

	# Crop masked_data
	crop = masked_data[y:y+h,x:x+w]
	return output, crop


def HSVBin(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

	lower_skin = np.array([100, 50, 0])
	upper_skin = np.array([125, 255, 255])

	mask = cv2.inRange(hsv, lower_skin, upper_skin)
	return mask


if __name__ == '__main__':
	main()
