# **Behavioral Cloning** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./write_up_img/nvifia_model.png "Model Visualization"
[image2]: ./write_up_img/steering_angle_distribution.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is same as the model that appeared in the Udacity lecture video. This model is a slight modified version of the NVidia team's model that was used to drive the real car. 

![alt text][image1]

At first, I tried designing my own model. Based on my previous experience with designing a CNN network and playing around with different model for this project, I quickly realized model architecture has very small impact on how well the car is trained to drive. As long a there are "enough" convolutions and "enough" dense layers, the models would perform very similarly. 

Although my model worked well (train and validation loss monotonically decreased) I chose to use the nvidia model because it had less parameters, thus had less chance of overfitting and used up less resource.

#### 2. Attempts to reduce overfitting in the model

I initially tried to train my model with some dropout layers. As expected, the dropout layers slowed down the training process. Since I was planning on using a lot of data, I figured dropout layers would slow down the training process even more and I can worry less about overfitting due to surplus training data.

I used separate training and validation data sets to train the network to see that each epoch wasn't over fitting. If the training loss monotonically decreased  with the validation loss and suddenly the validation loss increases relative to the previous epoch, I declared that epoch to be overfitting epoch and killed the training and restarted the training to train for 1 less than the overfitting epoch. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
As mentioned above, the network performance is not heavily dependent upon the model architecture. Since the model seemed to be pretty robust, the only parameter to "tune" was the number of epochs to train the network. 

#### 4. Appropriate training data

Since the model architecture and paramters did not affect the performance too much, my primary focus in training process was the quality and quantity of the data. In order to get the right "quantity" of data, I drove around the track to generate mroe data and performed few operations on the data to augment the data set. The quality of the data was also achieved by some pre-processing and recording data in specific locations. The details are below. 

### Data Collection and Processing details

#### 1. Data Collection

I think the most important part of this project was the data. The network performance increases as a function of data quality and quantity. At first, I followed the tips given in the lecture videos to collect data. I drove around the track in clockwise and counter clockwise directions. I stored these two sets of data in two separate folders called ./data_my_driving/clockwise and ./data_my_driving/counterclockwise. The given data was saved in ./data/data. As you can see, I completely separated the original given data from my own collected data because I wanted to play around with different data quality and easily be able to remove the unwanted data. For the same reason, I separated the ./data_my_driving directory into sub-directories of what kind of data I collected. 

The following is the list of data collection folders I created.

- **./data_my_driving/clockwise**
    - Normal driving around the track in clockwise direction
    - Tried to keep the car in the middle of the lane
- **./data_my_driving/counterclockwise**
    - Normal driving around the track in counterclockwise direction
    - Tried to keep the car in the middle of the lane
- **./data_my_driving/turns**
    - Recorded only turns (clockwise + counterclockwise)
- **./data_my_driving/sides**
    - Purposefully drove to the side and readjusted back to the center of the lane

I initially drove the car around the track in the counter clockwise direction. When I trained the network with this data, the car kept driving slightly towards the right and this small error accumulated to drive the car off the track. In order to adjust the biased angle towards the left, I drove the car around the track in opposite direction and saved the data in the ./data_my_driving/counterclockwise. 

After collecting data for clockwise and counterclockwise folders, I trained the netwrok and let it drive around the track. I performed very well on the straight roads but performed very poorly on some sharp turn. Also, it had a very difficult time at one point on the track where the right side of the road turns to mud. In order to better train the network, I data for only the truns. I drove various turns in both clockwise and counterclockwise directions. This helped the network perform much better on the turns.

I also tried to include data set for driving on the sides and readjusting, but this caused the car to drive in zig-zag manner. Therefore, I removed this dataset for training. 

#### 2. Augmenting data

For each frame, There are three camera views (Left, Center, Right). In order to utilize all three images, I had to offset the steering angle for the Left and Right camera images. I chose the value of 0.2 degrees because it was given during the lecture and it worked. For the left images, I added the correction angle and fro right camera images, I subtracted the value. This already created three labeled data per frame. In order to get more data, I took each image and flipped them along the y-axis and multiplied the angle by -1. The flipping of the image allows the car to have good balance of the clockwise and counterclockwise driving data, thus being able to center itself to the middle of the lane better. 

![alt text][image2]

#### 3. Data distribution

As seen in the image below, the steering angle distribution is heavily centered at zero degrees. This is due to two reasons.
1) Large portion of the track is staright, which does not required any steering.
2) When turning, the steering angle is not continuously offset to the left/right. The angle is steered towards the direction of the turn very shortly and it is followed by 0 degrees steering.

![alt text][image3]

The peaks around -0.2 and 0.2 degrees are due to the fact that data points for 0.2 and -0.2 degrees are derived from zero degree data. 

When there is overpopulated zero degree steering data, the car has a hard time making sharp turns. This is because the car's driving principal is biased towards zero degrees steering. My strategy to fix this issue was to limit the percentage of zero steering data to be included in the training. 

The parameter I used here was n_zero_max_percentage. If n_zero_max_percentage = 0.1, I would only allow 10% of the entire data to consist of zero degree data. I noticed that when this number is too low (little zero-degree data) the car wobbles around the track and cannot keep statble position at the center. This is because the car is trained to have very few zero degree steering. When this number is high, the car is more stable, but it has difficulties making sharp turn. In order to find the balance, I tried training the network with few different numbers and 0.3 worked the best in terms of visual performance. 

#### 4. Data pre-processing
The two pre-processing strategies I took are cropping the unnecessary top and bottom of the image and normalizing the image. The cropping helps in two ways: decreases the data size, thus decreasing the model complexity, and removes unnecessary information from disturbing the learning process.

#### 5. Driving speed
Although everything seemed like it should work, the car kept failing at certain turns. It seemed like the car was not turning fast enough. Then I realized the training data I took were collected at higher driving speed than the autonomous mode driving speed of 9. I increased the speed to 30 and the car drove around the track! However, the car seemed very unstable. I adjusted the speed to 20 mph and the car drove around the track with much more stability.

