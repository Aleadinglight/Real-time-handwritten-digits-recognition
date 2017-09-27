import cv2
import numpy as np
import predict

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades_eye.xml')
dest = np.random.randn(28,28)

while (True):
	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	height, width = gray.shape
	x = (width - height/3)/2
	crop = gray[height/3:height*2/3, x:x+height/3]
 	res = cv2.resize(crop, (28,28))

	mask = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 155, 1)
 	cv2.line(frame, (x,height/3),(x,height*2/3) , (0,0,0), 5)
 	cv2.line(frame, (x+height/3,height/3),(x+height/3,height*2/3) , (0,0,0), 5)

 	cv2.line(frame, (x,height/3),(x+height/15,height/3) , (0,0,0), 5)
 	cv2.line(frame, (x,height*2/3),(x+height/15,height*2/3) , (0,0,0), 5)
  	cv2.line(frame, (x+height/3, height/3),(x+height/3-height/15, height/3) , (0,0,0), 5)
	cv2.line(frame, (x+height/3, height*2/3),(x+height/3-height/15, height*2/3) , (0,0,0), 5)
 	
 	predict.predict(mask)
	cv2.imshow("frame",frame)
	cv2.imshow("crop",crop)
	cv2.imshow("mask",mask)
	#cv2.imshow("crop",crop)
	#cv2.imshow("frame",res)
	if (cv2.waitKey(30) & 0xff == ord('q')):
		break

cv2.destroyAllWindows()
cap.release()
