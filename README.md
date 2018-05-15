# **Traffic Sign Recognition** 


This is a brief writeup report of Self-Driving Car Engineer P2.

[//]: # (Image References)

[image1]: ./examples/german_traffic_sign.png
[image2]: ./examples/No.2_speed_limit_50.png
[image3]: ./examples/sign_label_bar.png
[image4]: ./examples/grayscale.png
[image5]: ./examples/validation_accuracy.png
[image6]: ./examples/additional_images.png
[image7]: ./examples/top5_prob.png

![alt text][image1]


---



## Data Set Summary & Exploration

### 1. Basic summary of the data set
Use the `pickle` library to load data and `.shape` to calculate summary statistics of the [German Traffic Signs](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `32, 32, 3`
* The number of unique classes/labels in the data set is `43`

### 2. Exploratory visualization of the dataset.

The German traffic signs data set has mapping relations between images and labels. Here is the image (with label index No.2) from the training data, and it's label is `Speed limit (50km/h)`.

![alt text][image2]

The data set has 43 unique classes, but the amount of each class is not even. The bar chart below shows how the data labels distributed.

![alt text][image3]


## Design and Test a Model Architecture

### 1. Preprocess the image data

Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

* **Convert the images to grayscale**

Use the `cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)` to convert images. 
Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

* **Normalize the image data**

Devide every pixel of an image by `255` to make the pixel value be in 0~1, which can approch the training process starting with a relatively good validation accuracy.

Normalize formulas tried:

| Formula | Validation accuracy | Test accuracy |
|:---:|:---:|:---:|
| (pixel - 128) / 128 | ~ 95% | ~ 93% |
| pixel / 255 - 0.5 | ~ 96% | ~ 94% |
| pixel / 255 | ~ 97% | ~ 95% |



* **Generate additional data**

    <font color=gray size=1>*Not yet.*</font>




### 2. Final model architecture
The final model consisted of the following layers:

| Layer | Description | 
|:---:|:---:|
| Input | 32x32x1 GRAY image | 
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x48 |
| RELU | |
| Max pooling | 2x2 stride,  outputs 14x14x48 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x96 |
| RELU | |
| Max pooling | 2x2 stride,  outputs 5x5x96 |
| Convolution 3x3 | 1x1 stride, valid padding, outputs 3x3x172 |
| RELU | |
| Max pooling | 1x1 stride,  outputs 2x2x172 |
| Fully connected | 688 input, output 240 |
| RELU | |
| Fully connected | 240 input, output 84 |
| RELU | |
| Fully connected | 84 input, output 43 |
| Softmax | |
 

### 3. Training model

By referencing the LeNet-5 model, the type of optimizer is AdamOptimizer, which used as `tf.train.AdamOptimizer` in TesorFlow. Here are 
hyperparameters set up in the training model:

* `EPOCHS = 50`
* `BATCH_SIZE = 128`
* `rate = 0.001`

### 4. The approach taken for finding a solution

During the training process(50 epoch), the accuracy changes as below：
![alt text][image1]

It has a nice start at about 80%, and increases beyond 90% very quickly. With the benefit from GTX970, the time overhead of training process is much less.

The final model results were:

* validation set accuracy of `97.6%`
* test set accuracy of `95.4%`

The neural networks model references the **LeNet-5** implementation with some layers and parameters fine tuned. LeNet-5 is a well known architecture, which chosen by these reasons:

* LeNet-5 is good at digit recognition, and it can be used in signs recognition similarly.
* The architecture is concise and elegant, which is beneficial to retrofit.
* Adjustable parameters are not many for training tune. 

BTW, at the beginning, when original LeNet-5 architecture and data set were used for training, the result accuracy was not so good (lower than 90%). Here is the approach taken for getting a better result:

* Convert the images to grayscale : Accuracy +2%
* Normalize the image data : Accuracy +1% 
* Add a 3x3 convolution layer and a Fully connected layer: Accuracy +4%
 

## Test a Model on New Images

### 1. Additional German traffic signs from web

Here are five German traffic signs that found on the web:

![alt text][image6]

The second image might be difficult to classify because the shape of sign is rather complicated and might be similar with pedestrains or children, which is hard to distinguish for the model when images grayscaled.

### 2. Model's predictions on new traffic signs

 and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image | Prediction | 
|:---:|:---:|
| Roundabout mandatory | Roundabout mandatory | 
| Dangerous curve to the right | Children crossing |
| Speed limit (70km/h) | Speed limit (70km/h) |
| General caution | General caution |
| Stop	 | Stop |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

### 3. The softmax probabilities for each prediction

Use `tf.nn.softmax(logits)` to calculate the probabilities and `tf.nn.top_k(prob, k=5)` to fetch the top 5 predictions.

Here are the top 5 softmax probabilities for each image along with the sign type of each probability. 

![alt text][image7]

For the first image, the model is relatively sure that this is a **Roundabout mandatory** sign (probability of 1.0), and the image does contain a roundabout mandatory sign. The top five soft max probabilities were:

| Probability | Prediction | 
|:---:|:---:|
| 1.00 | Roundabout mandatory | 
| 7.13e-15 | Speed limit (30km/h) |
| 1.36e-15 | Right-of-way at the next intersection |
| 1.47e-18 | Double curve |
| 5.18e-19 | Priority road |


For the second image, the model is relatively sure that this is a **Children crossing** sign (probability of 0.85), but the image does contain a Dangerous curve to the right sign. The top five soft max probabilities were:

| Probability | Prediction | 
|:---:|:---:|
| 0.85 | Children crossing | 
| 0.15 | Go straight or right |
| 1.67e-05 | Dangerous curve to the right |
| 5.68e-06 | Speed limit (60km/h) |
| 2.74e-06 | Bicycles crossing |


For the third image, the model is relatively sure that this is a **Speed limit (70km/h)** sign (probability of 1.0), but the image does contain a Dangerous curve to the right sign. The top five soft max probabilities were:

| Probability | Prediction | 
|:---:|:---:|
| 1.00 | Speed limit (70km/h) | 
| 4.68e-13 | Speed limit (30km/h) |
| 2.82e-15 | Speed limit (120km/h) |
| 2.65e-16 | Speed limit (100km/h) |
| 1.31e-17 | Speed limit (20km/h) |

For the forth image, the model is relatively sure that this is a **General caution** sign (probability of 1.0), and the image does contain a General caution sign. The top five soft max probabilities were:

| Probability | Prediction | 
|:---:|:---:|
| 1.00 | General caution | 
| 4.83e-31 | Go straight or left |
| 4.37e-31 | Pedestrians |
| 2.22e-37 | Traffic signals |
| 0.00 | Speed limit (20km/h) |

For the last image, the model is relatively sure that this is a **Stop** sign (probability of 1.0), but the image does contain a stop sign. The top five soft max probabilities were:

| Probability | Prediction | 
|:---:|:---:|
| 1.00 | Stop | 
| 5.50e-11 | Speed limit (120km/h) |
| 1.72e-12 | Speed limit (70km/h) |
| 1.07e-15 | No entry |
| 2.68e-16 | Keep left |