# Deep-learning-course-project
2019 Fall semester, Yonsei University

## Overview
What is the crowd counting, and why is it important?
Have you ever seen a news article concerning the number of people at political rallies? It has gone through a lot of controversy lately because the organizers and the opposite parties differed on the estimate of the crowd number. We can easily handle these problems by giving an objective estimate with a crowd counting machine. The domain of crowd counting can also be extended to other areas such as counting cells or bacteria from a microscopic image, counting animals in wildlife, or estimating the number of vehicles at transportation hubs.

## Data
The dataset we used is ‘Shanghai Tech dataset’ which is a popular, well organized dataset of crowd images.
In total, it consists of 1,198 images and about 330 thousands labeled people. It is divided into 2 parts, part A and part B. For part A, there are 182 test and 300 train images. For part B, it has 316 test images and 400 train images. Part A has more people per an image on average than Part B. It is because part A is random picture from the internet whereas part B is a picture taken in Shanghai street. We thought that by using a dataset divided into these two parts with different crowd densities, we can increase the accuracy of model prediction because we can consider the diversity of test data. We will also train our model separately for each part.

### Crowd Density Map 
In general, when we want to count the number of heads in an image, we use foreground segmentation. But in our case, the viewpoint of an image can be arbitrary, so it is hard to segment the crowd from its background without any information about geometry. Instead, we will use crowd density map. It is a matrix of how many people there are in per unit square area. Density map preserves more information. It also gives the spatial distribution of the crowd in the given image, so it is useful in many applications. Getting this matrix is easier than foreground segmentation, and once we have this matrix, we can easily estimate the number of people simply by summing up the matrix.

# Model
MCNN is multi-column convolutional neural network. It is an extended version of CNN. In MCNN, we stack several columns of this basic CNN. In our case, we used MCNN with three columns.
So why do we use MCNN not CNN? 
It is because in real life, the pictures of crowds are taken at very different angles. Our Shanghai Tech dataset also has images with various camera perspectives. 
Basic CNN, which consists of only one column cannot distinguish this difference of head sizes so it is ineffective for images like the right one. Therefore in our MCNN model, we used 3 different CNN columns for 3 different head sizes: large, medium, and small.

Now let’s talk about the structure of the model. The input is an color image of random size. 
This is the first CNN column for large head size,. For large heads, we use large filter size 9 by 9, and 7 by 7. We have 4 convolutional layers and three 2 by 2 max pooling layers. We used same padding and ReLU activation function.
For medium head, we used medium filter size which is 7 by 7 and 5 by 5.
For small head, we used small filter size which is 5 by 5 and 3 by 3.
And then, we merge these 3 outputs from different columns into one feature map.
Finally, we change this feature map into density map, which is what we want, by applying another convolutional layer of filter size 1 by 1.
After training, we plot the finally updated density map. 
Our model has over 413,000 parameters and you know, we have to go through the cnn process for three times. So it was impossible to train our model with our laptop CPUs so we used google colab.

In summary, our goal is to estimate a density map from an input image.
And then we can get the crowd number by integral of the density map.
 
# Experiments
Finally, let’s talk about our experiments.
We trained our model using 300 train data from part A and 400 train data from part B
Learning rate was 0.0001 and we used 200 epochs for each part. Like I have mentioned, we used google colab but it still took millions of times. For evaluation metric, we used MAE. MAE is mean absolute error and it indicates the accuracy of the estimates. The final MAE’s of test data after training was 128 for part A and 20.3 for part B. 

# Reference
Hajer Fradi, Jean-Luc Dugelay (2013) Crowd Density Map Estimation Based of Feautre Tracks

Shanghaitech University (2016) Single-Image Crowd Counting via Multi-Column Convolutional Neural Network 

Vihar Kurama (2019) Dense and Sparse Crowd Counting Methods and Techniques: A Review
https://nanonets.com/blog/crowd-counting-review/#crowd-counting-methods-and-techniques

Pulkit Sharma (2019)  A Must-Read Tutorial to Build your First Crowd Counting Model using Deep Learning
https://www.analyticsvidhya.com/blog/2019/02/building-crowd-counting-model-python/


