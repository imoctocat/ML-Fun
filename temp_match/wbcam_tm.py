import cv2 
import numpy as np

# Reading video feed
video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()



template = cv2.imread('eye.png',0)

while True:
	ret, frame = video_capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	w, h =template.shape[::-1]
	print(template.shape,"Shape")
	
	## Performing tempelate match
	res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
	threshold = 0.61
	loc = np.where( res>= threshold) # filtering out results with low thresold
	print(loc)
	## Creating bounding boxes on those areas which pass the threshold value
	for pt in zip(*loc[::-1]):
		cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
	##Displaying the ouput
	cv2.imshow('Detected', frame)
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

