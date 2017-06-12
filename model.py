import csv
import os

import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

import time
# model.py

# TODO LIST
# - Take in an image from the center camera of the car. This is the input to your neural network.
# - Output a new steering angle for the car.

current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path_list = [os.path.join(current_file_path,"data","data")]#, os.path.join(current_file_path,"data_my_driving")]



def load_data(data_path):
    samples = []
    csvpath = os.path.join(data_path,"driving_log.csv")
    with open(csvpath) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            for ii,l in enumerate(line[:3]):
                line[ii] = os.path.join(data_path,"IMG",os.path.basename(l))
                line_mod = [line[ii]] + line[3:]
                samples.append(line_mod)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples


train_samples = []
validation_samples = [] 
for data_path in data_path_list:
    train_samples_tmp, validation_samples_tmp = load_data(data_path)
    train_samples += train_samples_tmp
    validation_samples += validation_samples_tmp

print(len(train_samples))

def generator(samples, angle_correction=0.2,batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            t_read = 0
            for batch_sample in batch_samples:
                path = batch_sample[0]
                angle = batch_sample[1]
                t0 = time.time()
                center_image = cv2.imread(path)
                t_read+=(time.time() - t0)
                center_angle = float(angle)
                if "left_" in path:
                    center_angle += angle_correction
                elif "right_": # right
                    center_angle -= angle_correction
                images.append(center_image)
                angles.append(center_angle)
            
            print(t_read)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
batch_size = 256
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


next(train_generator)

ch, row, col = 3, 80, 320  # Trimmed image format

crop_dim = ((50,20),(0,0)) # ((crop_top, crop_bot) , (crop_left,crop_right))
input_shape = (160,320,3)
# ---- import from keras ----
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Input, Flatten, Dense, Lambda
from keras.layers import Activation, Dropout
from keras.layers import Conv2D,MaxPooling2D
# ---------------------------

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
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1.))

    # === build model ===
    # conv layer 1
    model.add(Conv2D(16,(5,5),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv layer 2
    model.add(Conv2D(32,(5,5),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # conv layer 3 
    model.add(Conv2D(64,(5,5),activation='relu'))
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


    model.compile(loss='mse', optimizer='adam')

    if print_shape: model_printer(model)
        
    return model


def simple_model_builder(input_shape,print_shape=True):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    if print_shape: model_printer(model)
        
    return model


# model = model_builder(crop_dim,input_shape)


model = simple_model_builder(input_shape)
# model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
#     validation_data=validation_generator, \
#     nb_val_samples=len(validation_samples), nb_epoch=3)