## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# **Traffic Sign Recognition** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/count_plot.png "Visualization"
[image2]: ./examples/random_16.png "Random examples"
[image3]: ./examples/dark_images.png "Dark images"
[image4]: ./examples/dark_example.png "Dark example"
[image5]: ./examples/clahe_example.png "CLAHE example"
[image6]: ./examples/original_flip.png "Original flip"
[image7]: ./examples/flipped.png "Flipped"
[image8]: ./examples/test_images.png "Test images"
[image9]: ./examples/test_results.png "Test results"
[image10]: ./examples/feature_maps.png "Feature maps"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it! and here is a link to my [project code](https://github.com/KalleBylin/SDCND-German-Traffic-Signs/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34.799 images
* The size of the validation set is 4.410 images
* The size of test set is 12.630 images
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

It is always a good idea to do an exploratory visualization of the data set to understand what we are working with before jumping in. One of the first visualizations I made was a bar chart showing the distribution of the classes in the training data as can be seen here:

![count plot][image1]

This simple step shows us that there is quite an obvious degree of imbalance in the number of examples between classes which we definitely look into to avoid bias towards the majority classes.

I also plotted out 16 random images from the dataset:

![random 16][image2]

A small detail that jumps out here is the brightness of the images. It definitely seems like some images are either very bright or very dark. Further investigation shows that about a third of the training images have an average brightness of 60 or less in a range from 0 to 255. Here we can see a few random examples from this dark third of the training set:

![dark images][image3]

As we can see above, some of the images are so dark that it becomes hard for even a human to classify the images. This is definitely something we will look into during the data preprocessing step.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


As I just mentioned, some images are very difficult for even a human to identify. So one of the first processing steps I focused on was to deal with this issue. We have different ways of dealing with this problem like histogram equalization. This is a technique that we use to adjust image intensities in order to enhance contrast by analyzing the intensity values and spreading out the most frequent intensities. This method has a well-known limitation though when some areas of the image are much darker or brighter than the rest. This is because when the intensities of the whole image are distributed these areas are usually still too dark or bright. To solve this, it is common to use a mor localized approach with adaptive histogram equalization which computes several histograms corresponding to different sections of the image.

We will use a method that goes one step further called Contrast Limited Adaptive Histogram Equalization (CLAHE), because the adaptive histogram equalization sometimes tends to overamplify noise in areas of the image that are relatively homogeneous. The CLAHE method tries to reduce this overamplification of noise by clipping the histogram at a predefined value.


I also decided to convert the images to grayscale because most of the traffic signs in this dataset are visually quite distinctive without taking into account the color. Grayscaling allows us to create a much smaller network and avoid using as many parameters. This helps our model to generalize better with less data. When we add complexity to the process we usually end up needing more data.

Here is an example of a dark traffic sign image:

![dark example][image4]

And here is the same example after grayscaling + Contrast Limited Adaptive Histrogram Equalization:

![clahe example][image5]

Before the processing step it isn't easy to identify which traffic sign is in the image, but we can clearly see the form and content of the traffic sign after applying CLAHE. 

Addtional to this I decided to normalize the data. Even though, strictly speaking, numeric data doesn't have to be normalized when working with neural networks, experience has shown that it really does make learning more efficient. The pixel values are in the range 0 to 255 by default so I subtracted 128 to each pixel and divided by 128.

I also created additional data. Looking at examples from each traffic sign it was quite clear that several of the traffic signs could be flipped horizontally and some signs were mirrored versions of others (e.g. "Keep right" vs "Keep left"). So the first type of images were mirrored to double the amount of data for that particular traffic sign, and for the second type I added the mirrored images of a traffic sign to its counterpart.

Here we can see an example of a "Keep left" sign:

![original flip][image6]

And after flipping the image it effectively becomes a "Keep right" sign:

![flip example][image7]

This step added approximately 14.000 images to our training data. Further data augmentation was done during training including vertical and horizontal shifting and zooming of the images.  



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.

My final model was loosely inspired on the VGG architecture. The VGG blocks usually contain multiple stacked smaller-sized kernels (usually 3x3) due to the intuition that this is better than larger-sized kernel. The idea here is that this allows the network to to learn more complex features at a lower cost due to multiple non-linear layers that increase the depth of the network. The kernel size of my convolutional layers was 5x5 instead of 3x3 but I decided to create three blocks of two stacked convolutional layers followed by Max pooling, dropout and a Relu activation.

After this I concatenated the output of the three blocks before flattening to generate skip connections. These are connections that skip one or more layers and helps us to break symmetries that can cause the model to learn slower.  

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input             | 32x32x1 Grayscale image   			 		    | 
| Convolution 5x5   | 1x1 stride, same padding, outputs 32x32x32 	    |
| Convolution 5x5   | 1x1 stride, same padding, outputs 32x32x32    	|
| Max pooling	   	| 2x2 stride,  outputs 16x16x32 	    			|
| Dropout           |									     			|
| RELU		        |								    				|
| Convolution 5x5   | 1x1 stride, same padding, outputs 16x16x64 	    |
| Convolution 5x5   | 1x1 stride, same padding, outputs 16x16x64 	    |
| Max pooling	    | 2x2 stride,  outputs 8x8x64 				        |
| Dropout           |									     			|
| RELU		        |									     			|
| Convolution 5x5   | 1x1 stride, same padding, outputs 8x8x128     	|
| Convolution 5x5   | 1x1 stride, same padding, outputs 8x8x128 	    |
| Max pooling	   	| 2x2 stride,  outputs 4x4x128 			    	    |
| Dropout           |									     			|
| RELU				|											    	|
| Concatenation  	| Connects output of three blocks, outputs 4x4x224  |
| Flatten           | outputs (3584,)                                   |
| Fully connected	| outputs (1024,)        							|
| Dropout           |									     			|
| Fully connected	| outputs (512,)        							|
| Dropout           |									     			|
| Softmax		    | outputs (43,)   									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the `ImageDataGenerator`¨class from Keras to easily generate batches of images with real-time augmentation. This was done to reduce the of first augmenting the data and keeping it all in memory.

I trained the model for 50 epochs with a batch size of 128 and a learning rate of 0.001. The optimizer used was the Adam optimizer which is very efficient and works very well in a wide range of tasks. It adapts the training rate during training by, in short terms, calculating an exponential moving average of the gradient and the squared gradient.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model had a validation set accuracy of 97.2%, compared to the 0.89-0.90% accuracy of the LeNet-5 model.

The whole process was very iterative. I first started by just tweaking the default LeNet-5 architecture by making it more complex adding more channels to each convolutional layer but this didn't help much. The architecture was still quite simple and kept underfitting. It was at this point that I decided to make the problem easier by switching over to grayscale images. This helped but I still wasn't reaching the desired accuracy.

At this point I started looking at more complex deep learning architectures for inspiration and for other ways of improving my data during preprocessing. So I ended up using CLAHE to address the brightness issue and I created blocks of stacked convolutional layers inspired by the blocks found in VGG architectures allowing me to create a deeper model. To reduce the risk of overfitting I added dropout as well.

I experimented with different learning rates and adding my own learning rate decay, but in the end the initial 0.001 had the best results. I also went from 10 to 50 epochs as the model was clearly still learning when hitting 10 epochs. 



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I decided to go with 9 German traffic signs that I found on the web:

![alt text][image8]

These images are not very unlike the ones found in the dataset. Some of them are slightly shifted in terms of perspective and they have all been scaled down to 32x32.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield					| Yield											|
| Stop Sign      		| Stop sign   									| 
| Slippery Road			| Slippery Road      							|
| No entry     			| No entry 										|
| Keep left 			| Keep left										|
| Bumpy road      		| Bumpy Road					 				|
| Keep right 			| Keep right									|
| 70 km/h	      		| 70 km/h	         			 				|
| No passing   			| No passing									|



The model was able to correctly guess all 9 of the traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.5%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.


![alt text][image9]

Above we can see each of the 9 images followed by a plot with the top five probabilities. For most images the top 1 probability is practically 1. The most unsure predictions are:
- "Keep right" which chose the correct class with a probability of about 0.8.
- "Slippery road" with a top 1 probability of almost 0.5
- "Keep left" was the least certain prediction with the following classes and approximate probabilities: "Keep Left" - 0.15, "Priority road" - 0.14, "30 km/h" - 0.08, "No vehicles" - 0.08 and "No passing" - 0.06. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image10]

This visualization is quite interesting as it shows how each feature map focuses on different features. For example feature map 16 seems to pick up on vertical lines while others (like feature maps 10, 11, 23 and 24) focus on diagonal lines in different directions. Practically all of them focus on what is happening in the center of the image, ignoring the background of the signs. There are also some feature maps that seem to pick up on the characters found inside of the sign.
