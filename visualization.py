# visualization.py
from model import *
import matplotlib.pyplot as plt 
from copy import deepcopy

def getGenerator(n):
	current_file_path = os.path.dirname(os.path.realpath(__file__))
	data_path_list = [os.path.join(current_file_path,"data","data")]#, os.path.join(current_file_path,"data_my_driving")]
	LOAD_IMAGE=True
	train_samples,validation_samples = load_data_multiple_paths(data_path_list)
	print(train_samples[:2])
	train_generator = generator(train_samples, batch_size=n,SHUFFLE=False)

	return train_generator


def compareOriginalImage_and_flip():
	gen = getGenerator(2)
	x,y = next(gen)
	x_original, y_original = x[0],y[0]
	x_flip, y_flip = x[1],y[1]


	plt.figure(figsize=[10,5])
	plt.subplot(1,2,1)
	plt.imshow(x_original[:,:,::-1])
	plt.axis('off')

	plt.subplot(1,2,2)
	plt.imshow(x_flip[:,:,::-1])
	plt.axis('off')

def main():
	compareOriginalImage_and_flip()
	plt.show()


if __name__ == '__main__':
	main()