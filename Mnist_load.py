import os
import numpy as np
from PIL import Image
from scipy import ndimage
from matplotlib import pyplot as plt
import cv2
import pickle

def load():
	data = []
	# this is the directory of the data set
	data_path="/home/tamnguyen/Documents/Mnist_Pytorch/mnist_png/training"
	# os.getcwd : get the currently working directory
	# List all the test types from 'A' - 'Z'
	test_type = os.listdir(data_path)
	# Now iterate through all test types
	i=0
	for each_type in test_type:
		# Specify the path to the folder contain all test type *each_type*
		type_path = data_path+"/"+each_type;
		# List all test set in this type
		test_set = os.listdir(type_path)
		# Pick out all tests of this type and put it in out data
		for each_test_set in test_set:
			i=i+1
			img_path = type_path + "/" + each_test_set
			img = ndimage.imread(img_path).astype(float)
			img = img/255
			each_type = int(each_type)
			tmp = (img, each_type)
			data.append(tmp)
		print "Done preprocess data type: ", each_type
		
	filename = open("mnist_data.dat","wb")
	pickle.dump(data, filename)
	filename.close()
	print "Done saving",i,"data sets into file: mnist_data.dat"
	#cv2.imshow("img",img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

if __name__ == "__main__":
	load()