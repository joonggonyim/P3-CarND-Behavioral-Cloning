#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is same as the model that appeared in the Udacity lecture video. This model is a slight modified version of the NVidia team's model that was used to drive the real car. 

![alt text][image1]

At first, I tried designing my own model. Based on my previous experience with designing a CNN network and playing around with different model for this project, I quickly realized model architecture has very small impact on how well the car is trained to drive. As long a there are "enough" convolutions and "enough" dense layers, the models would perform very similarly. 

Although my model worked well (train and validation loss monotonically decreased) I chose to use the nvidia model because it had less parameters, thus had less chance of overfitting and used up less resource.

####2. Attempts to reduce overfitting in the model

I initially tried to train my model with some dropout layers. As expected, the dropout layers slowed down the training process. Since I was planning on using a lot of data, I figured dropout layers would slow down the training process even more and I can worry less about overfitting due to surplus training data.

I used separate training and validation data sets to train the network to see that each epoch wasn't over fitting. If the training loss monotonically decreased  with the validation loss and suddenly the validation loss increases relative to the previous epoch, I declared that epoch to be overfitting epoch and killed the training and restarted the training to train for 1 less than the overfitting epoch. 

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
As mentioned above, the network performance is not heavily dependent upon the model architecture. Since the model seemed to be pretty robust, the only parameter to "tune" was the number of epochs to train the network. 

####4. Appropriate training data

Since the model architecture and paramters did not affect the performance too much, my primary focus in training process was the quality and quantity of the data. In order to get the right "quantity" of data, I drove around the track to generate mroe data and performed few operations on the data to augment the data set. The quality of the data was also achieved by some pre-processing and recording data in specific locations. The details are below. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
