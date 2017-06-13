# visualization.py
from model import *
import matplotlib.pyplot as plt 
from copy import deepcopy

def getGenerator(data_path_list,n_zero_max_percentage,n=1,SHUFFLE=False):
	# current_file_path = os.path.dirname(os.path.realpath(__file__))
	# data_path_list = [os.path.join(current_file_path,"data","data")]#, os.path.join(current_file_path,"data_my_driving")]


	train_samples,validation_samples = load_data_multiple_paths(data_path_list,test_size=None,n_zero_max_percentage=n_zero_max_percentage)
	n_samples = len(train_samples) + len(validation_samples)
	if not n:
		n = n_samples
	print(n)
	train_generator = generator(train_samples, batch_size=n,SHUFFLE=SHUFFLE)

	return train_generator,n_samples


def compareOriginalImage_and_flip():
	current_file_path = os.path.dirname(os.path.realpath(__file__))
	data_path_list = [os.path.join(current_file_path,"data","data")]#, os.path.join(current_file_path,"data_my_driving")]
	ii_rand = np.random.randint(1000)
	ii_rand += ii_rand%2
	gen,n_samples = getGenerator(data_path_list,n=ii_rand)

	x,y = next(gen)

	x_original, y_original = x[-2],y[-2]
	x_flip, y_flip = x[-1],y[-1]


	plt.figure(figsize=[10,5])
	plt.subplot(1,2,1)
	plt.imshow(x_original[:,:,::-1])
	plt.axis('off')
	plt.title("Original Image \nAngle of turn : {}".format(y_original))

	plt.subplot(1,2,2)
	plt.imshow(x_flip[:,:,::-1])
	plt.axis('off')
	plt.title("Flipped Image \nAngle of turn : {}".format(y_flip))

def plotSteerAngleDistribution(angle_correction=0.2):
	current_file_path = os.path.dirname(os.path.realpath(__file__))
	data_path_list = [os.path.join(current_file_path,"data","data"), 
                      os.path.join(current_file_path,"data_my_driving",'clockwise'),
                      os.path.join(current_file_path,"data_my_driving",'counterclockwise'),
                      os.path.join(current_file_path,"data_my_driving",'turns')]


	n = 1000 
	n_zero_max_percentage = 0.1
	gen_nolimit,_ = getGenerator(data_path_list,n_zero_max_percentage=1,n=n,SHUFFLE=True)
	gen_limit,_ = getGenerator(data_path_list,n_zero_max_percentage=n_zero_max_percentage,n=n,SHUFFLE=True)

	_,y_nolimit = next(gen_nolimit)
	_,y_limit = next(gen_limit)


	plt.figure(figsize=[10,8])

	plt.subplot(2,1,1)
	plt.title('Include All data')
	plt.hist(y_nolimit,31,edgecolor='k')
	
	plt.subplot(2,1,2)
	plt.title("Include {}% of zero-degree data".format(n_zero_max_percentage*100))
	plt.hist(y_limit,31,edgecolor='k')
	plt.xlabel("Steer angle (deg)")

	plt.savefig("./writeup_img/steering_angle_distribution.png")
def main():
	# compareOriginalImage_and_flip()
	plotSteerAngleDistribution()
	plt.show()


if __name__ == '__main__':
	main()