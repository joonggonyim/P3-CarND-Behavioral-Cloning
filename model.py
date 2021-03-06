import csv
import os

import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
# ---- import from keras ----
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Input, Flatten, Dense, Lambda
from keras.layers import Activation, Dropout
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D
import keras.backend as K
# ---------------------------
import time
# model.py

# TODO LIST
# - Take in an image from the center camera of the car. This is the input to your neural network.
# - Output a new steering angle for the car.



def load_data(data_path,test_size,n_zero_max_percentage):
    samples = []
    csvpath = os.path.join(data_path,"driving_log.csv")
    n_zero = 0
    n_samples = 0
    with open(csvpath) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            # assume you are using the following 6 samples (center,left,right) and flipped of these
            n_samples+=6
            angle = float(line[3])
            if angle == 0:
                # check if the number of zeros exceeded
                # print(n_zero/n_samples , n_zero_max_percentage)
                if n_zero/n_samples <= n_zero_max_percentage:
                    n_zero+=6
                    # loop through center,left,right images
                    for ii,l in enumerate(line[:3]):
                        

                        line[ii] = os.path.join(data_path,"IMG",os.path.basename(l))
                        # for each line, add a flag that indicates if image should be flipped
                        line_mod = [line[ii]] + line[3:] + ['original']
                        line_mod_flip = [line[ii]] + line[3:] + ['flip']
                        samples.append(line_mod)
                        samples.append(line_mod_flip)
                else:
                    # the number of zeros exceeded, decrement the number of sample back by -6
                    n_samples-=6
            else:
                # angle is not zero
                # loop through center,left,right images
                for ii,l in enumerate(line[:3]):
                    

                    line[ii] = os.path.join(data_path,"IMG",os.path.basename(l))
                    # for each line, add a flag that indicates if image should be flipped
                    line_mod = [line[ii]] + line[3:] + ['original']
                    line_mod_flip = [line[ii]] + line[3:] + ['flip']
                    samples.append(line_mod)
                    samples.append(line_mod_flip)

    if test_size and test_size > 0:
        train_samples, validation_samples = train_test_split(samples, test_size=test_size)
        return train_samples, validation_samples
    else :
        return samples,[]

    

def load_data_multiple_paths(data_path_list,test_size=0.2,n_zero_max_percentage=0.1):

    train_samples = []
    validation_samples = [] 
    for data_path in data_path_list:
        train_samples_tmp, validation_samples_tmp = load_data(data_path,test_size,n_zero_max_percentage)
        train_samples += train_samples_tmp
        validation_samples += validation_samples_tmp

    return train_samples,validation_samples



def generator(samples,angle_amplifier=1, angle_correction=0.2,batch_size=32,SHUFFLE=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates   

        if SHUFFLE:
            sklearn.utils.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            t_read = 0
            for batch_sample in batch_samples:
                path = batch_sample[0]
                angle = batch_sample[1]
                FLIP = (batch_sample[-1] == 'flip')
                center_image = cv2.imread(path)
                center_angle = float(angle)
        
                if "left_" in path:
                    center_angle += angle_correction
                elif "right_" in path: # right
                    center_angle -= angle_correction

                if FLIP:
                    center_image = cv2.flip(center_image,1)
                    center_angle = -1*center_angle

                center_angle = center_angle*angle_amplifier
                
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            
            if SHUFFLE:
                yield sklearn.utils.shuffle(X_train, y_train)
            else:
                yield X_train, y_train



def model_printer(model):
    line_breaker = "+{:-<22}+{:-<22}+".format("","")
    fmt = "| {:^20} | {:^20} |"

    print(line_breaker)    
    print(fmt.format("Layer Name","Shape"))
    print(line_breaker)
    for layer in model.layers:
        list_str = ','.join(str(e) for e in layer.output.shape.as_list())
        print(fmt.format(layer.output.name.split("/")[0], list_str))
    print(line_breaker)

def model_builder(crop_dim,input_shape,print_shape=True):
    model = Sequential()
    # === Pre-processing data ===
    # Crop image
    model.add(Cropping2D(cropping=crop_dim, input_shape=(160,320,3)))
    # Averagepool tp decrease the data size
    model.add(AveragePooling2D(pool_size=(2,2)))
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1.))

    # === build model ===
    # conv layer 1
    model.add(Conv2D(8,(5,5),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv layer 2
    model.add(Conv2D(16,(5,5),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # Flatten the 4d tensor
    model.add(Flatten())

    # FC 1
    model.add(Dense(5000,activation='relu'))
    model.add(Dropout(0.25))

    # FC2 
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))

    # FC3
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.25))

    # FC4
    model.add(Dense(1))
    model.add(Dropout(0.25))

    # # limits the final angle between the pre_defined range to prevent overshoot
    # model.add(Lambda(lambda x: K.maximum(x,angle_range[0])))
    # model.add(Lambda(lambda x: K.minimum(x,angle_range[1])))



    model.compile(loss='mse', optimizer='adam')

    if print_shape: model_printer(model)
        
    return model


def model_builder_nvidia(crop_dim,input_shape,print_shape=True):
    model = Sequential()
    model.add(Cropping2D(cropping=crop_dim, input_shape=input_shape))
    model.add(Lambda(lambda x: x/127.5 - 1.))

    model.add(Conv2D(8,(5,5),strides=(2,2),activation='relu'))
    model.add(Conv2D(36,(5,5),strides=(2,2),activation='relu'))
    model.add(Conv2D(48,(5,5),strides=(2,2),activation='relu'))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Conv2D(64,(3,3),activation='relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')

    if print_shape: model_printer(model)
        
    return model
    

def main():
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    data_path_list = [os.path.join(current_file_path,"data","data"), 
                      os.path.join(current_file_path,"data_my_driving",'clockwise'),
                      os.path.join(current_file_path,"data_my_driving",'counterclockwise'),
                      os.path.join(current_file_path,"data_my_driving",'turns')]
                      # os.path.join(current_file_path,"data_my_driving",'sides')]

    n_zero_max_percentage = 0.5 # 67844
    train_samples,validation_samples = load_data_multiple_paths(data_path_list,n_zero_max_percentage=n_zero_max_percentage)



    # compile and train the model using the generator function
    batch_size = 128
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)


    # next(train_generator)

    ch, row, col = 3, 80, 320  # Trimmed image format

    crop_dim = ((70,25),(0,0)) # ((crop_top, crop_bot) , (crop_left,crop_right))
    input_shape = (160,320,3)

    model = model_builder_nvidia(crop_dim,input_shape)    

    # model = model_builder(crop_dim,input_shape)



    model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size), \
        validation_data=validation_generator, \
        validation_steps=np.ceil(len(validation_samples)/batch_size), nb_epoch=3)

    model.save('model.h5')


if __name__ == '__main__':
    main()