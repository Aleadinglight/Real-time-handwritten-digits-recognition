import random
import cv2
import numpy as np 
import argparse
import pickle

parser = argparse.ArgumentParser(description="Data checking")
parser.add_argument("-f","--filename", type = str)

args = parser.parse_args()

def check(file_name):
	print file_name
	with open(file_name,"rb") as file:
		data = pickle.load(file)
	#data is a list of tuple (28x28, 1) meaning the picture 28x28 and the coresponding class 0-9
	print np.shape(data)
	n = len(data)
	while True:
		i = random.randint(0, n)
		print data[i][1]
		cv2.imshow("picture",data[i][0]);
		if (cv2.waitKey(0) & 0xFF == ord('q')):
			cv2.destroyAllWindows()
			break
		else:
			cv2.destroyAllWindows()

		
if __name__ == "__main__":
	check(args.filename)