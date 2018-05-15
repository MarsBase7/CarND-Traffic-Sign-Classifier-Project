# **Traffic Sign Recognition** 


This is a brief writeup report of Self-Driving Car Engineer P2.

[//]: # (Image References)

[image1]: ./examples/german_traffic_sign.png
[image2]: ./examples/No.2_speed_limit_50.png
[image3]: ./examples/sign_label_bar.png
[image4]: ./examples/grayscale.png

![alt text][image1]


---



## Data Set Summary & Exploration

#### Basic summary of the data set
Use the `pickle` library to load data and `.shape` to calculate summary statistics of the [German Traffic Signs](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `32, 32, 3`
* The number of unique classes/labels in the data set is `43`

#### Exploratory visualization of the dataset.

The German traffic signs data set has mapping relations between images and labels. Here is the image (with label index No.2) from the training data, and it's label is `Speed limit (50km/h)`.

![alt text][image2]

The data set has 43 unique classes, but the amount of each class is not even. The bar chart below shows how the data labels distributed.

![alt text][image3]


## Design and Test a Model Architecture

#### preprocess the image data

Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

**1. Convert the images to grayscale**

Use the `cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)` to convert images. 
Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

**2. Normalize the image data**

Devide every pixel of an image by `255` to make the pixel value be in 0~1, which can approch the training process starting with a relatively good validation accuracy.

Normalize formulas tried:

| Formula | Validation accuracy | Test accuracy |
|:---:|:---:|:---:|
| (pixel - 128) / 128 | ~ 95% | ~ 93% |
| pixel / 255 - 0.5 | ~ 96% | ~ 94% |
| pixel / 255 | ~ 97% | ~ 95% |



**3. Generate additional data**

<small>Not yet.</small>




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

## Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 



