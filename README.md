# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Instructions for the project can be found [here](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/README.md)

The rubric for the project can be found [here](https://review.udacity.com/#!/rubrics/481/view)

## Data Set Summary & Exploration

### 1. Provide a basic summary of the data set.

Here's a brief summary of the data set

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

### 2. Include an exploratory visualization of the dataset.

The distribution of data across classes for the training set is show below.

![alt text][image1]

As you can see, there is a significant disparity in the number of images across classes. This hints at the need for supplementation of images across classes to narrow down this gap, otherwise, the model may be biased towards the signs with the most samples. Obviously, some signs are more rare than others, but in the case of driving, it wouldn't be acceptable to simply fail on rare signs; the classifier should ideally classifly all signs correctly.


Below is a sampling of images from the training set.

![alt text][image2]

The most noticeable aspect of the images here is that they vary greatly in luminosity. This is to be expected, as self-driving cars can't be expected to only operate in the day time. This hints at the need for some sort of luminosity correction in the image processing pipeline.

## Design and Test a Model Architecture

### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?

The two factors that played the largest role in determining an appropriate image processing pipeline were:
1. The large disparity in the number of samples across classes - as shown in the section above - which has the potential to bias the NN during training, and
2. The vast differences in luminosity across samples - also illustrated in the section above.

To deal with the issues above, I set up an augmentation pipeline to create additional images. For every class, the augmentation pipeline:
- Determines the number of additional images to be created. It seemed appropriate to narrow the disparity across classes, and as such, I chose to create 5x the number of samples in each class, with a minimum limit of 1000 and a maximum limit of 3000 samples.
-Per additional image, selects an image to be augmented at random from the original set.
- Modifies the selected image by applying randomized rotation, blurring, and affine warping.

The figure below illustrates the effects of the augmentation pipeline on a single sample image.

![alt text][image3]

All augmentation was performed on the raw data set, and the augmented set was saved. The distribution of images across classes after augmentation is shown below.

![alt text][image4]

The histogram below shows the distribution of images after supplementation.

![alt_text][image5]

To deal with 2), a luminosity correction was performed by converting the image to YUV space, and applying a localized Clahe histogram equalization on the Y layer. The images were then converted to grayscale. The YUV correction and conversion to grayscale also match the approach taken by [Sermanet & LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf); regarding the grayscale conversion specifically, the authors noted that using color images didn't increase predication accuracy. The figure below shows the effects of image processing on the same set of images presented in **Dataset Summary and Exploration** section.

![alt text][image6]


### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I based my on architecture on that proposed by [Sermanet & LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). Fankly, I didn't have the processing power (or time) to run NNs with the depth that produced the best results in this paper, so I selected the following architecture:

**Stage 1**
- Input Layer - 32x32x3 RGB image
- Convolutional Layer 1 - 5x5 convolution, 1x1 stride, valid padding; output 28x28x10
- Relu + Dropout 1
- Max Pool 1 - 2x2 filter with 2x2 stride, valid padding; output 14x14x10

**Stage 2**
- Convolutional Layer 2 - 5x5 convolution, 1x1 stride, valid padding; output 10x10x20
- Relu + Dropout 2
- Max Pool 2 - 2x3 filter with 2x2 stride, valid padding; output 5x5x10

**Flattening**
- Convolutional Layer 3 - 5x5 convolution, 1x1 stride, valid padding; output 1x1x500 (i.e.: flattening convolution)
- Dropout 2
- Flatten output of stage 1 (14x14x10); output 1960
- Concatentate with output of Convolutional Layer 3 (after dropout); output 2460

**Classifier**
- Connected Layer 1 - 2460 x 120
- Relu + Dropout 3
- Connected Layer 2 - 120 x 84
- Relu + Dropout 4
- Connected Layer 3 - 84x43

The classifier is based on LeNet with adjustments made to account for increased matrix sizes. Four different rates of dropout were applied to different layers in the architecture; this was based on findings that high rates of dropout in the largest connected layers were beneficial, but those same high rates of dropout in the convolutional and smaller connected layers were detrimental.



### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam Optimizer used in the Udacity classroom sessions. I experimented with various parameters, but settled on the following:

- **Epochs** - 6
    - Generally, the network appeared to plateau in accuracy within the first 5-7 epochs - as shown in the learning curve below - and would fluctuate afterwards. I tried to vary the amounts of dropout in the network to cause training and validation accuracy converge (i.e.: preven overfitting), but this was about as close as I could get. Based on the curve below, I kept the number of epochs at 6 to prevent overtraining.

![alt text][image10]

Note that the learning curve shown in the jupyter notebook contains fewer epochs; the above curve was generated first to determine the number of epochs to run.

- **Batch Size** - 128
    - I didn't experiment much with this.

- **Learning Rate** - 0.0008
    - Higher learning rates tended to result in lower accuracy, so I decreased the learning rate to the point that accuracy appeared to plateau.

- **Matrix Initialization** - Mean - 0.0, Std Dev - 0.1
    - Standard deviation of random matrices seemed to dramatically affect the rates of convergence across epochs. I tested standard deviations of several orders of magnitude; lower standard deviations (ex: 0.01) tended to result in slow convergence to slightly lower levels of accuracy (ex: 95%), whereas higher standard deviations (ex: 0.5) would typically result in very poor accuracy. I'm assuming that the larger standard deviation created an "opinionated" network upon initialization.

- **Dropout**
  - I experimented with various levels of dropout at different layers in the architecture. I provided the ability to apply different levels of dropout across the layers, but ultimately, I included dropout only in larger layers, with convolutional layers having less dropout than connected layers. Including high rates of dropout in the convoluational layers or smaller connected layers tended to decrease accuracy. Intuitively, this makes sense: high levels or dropout in areas of the network where the number of parameters is small (such as the later convolutional layers) means that you're "throwing out" too much information. In the future, if I had more processing power, I might try to add more convolutional depth (ex: 108 features, as suggested by [Sermanet & LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)) and increase dropout levels.


### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Performance of the network is as follows:
- Validation set accuracy of: ~97.5%
- Test set accuracy of: ~96.0%

Much of the decision making in arriving at this architecture with the given hyperparameters is described in the sections above. In summary, I chose an architecture based on that presented by [Sermanet & LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), but chose to add less depth than suggested due to processing and time constraints. Aside from the effects of the hyper parameters detailed above, two aspects of the overally processing and classification pipeline provided the largest benefit:

1. **Data Set Augmentation**
    - Evening out the disparity in the number of images across classes appeared add roughly 1% accuracy
    - Note: in the future, it would make sense to analyze the performance of the classifier per image label to determine where augmentation could or does provide the greatest benefit.

2. **Luminosity Correction**
    - Correcting luminosity in the images added roughly 2% accuracy; this appeard to be the single largest factor affecting accuracy aside from selecting the correct parameters for the pipeline.


## Test a Model on New Images

### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Below are 6 German Traffic signs I found on the web, before and after they were run through the image processing pipeline:

![alt text][image7]

I chose 3 images of the same shape (triangle with red border) to test how well the network was able to distinguish the elements in the center of the sign. The others were chosen at random. In general, I felt all should be rather simple to classify.


### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Answers to 2 and 3 combined below.

### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The results of the prediction as well as the top softmax probabilities are shown in the image below.

![alt text][image8]

**Double Curve**
The classifier was unable to classify the double curve sign correctly, predicting that it was "children crossing" with . The originial image I selected from the web was actually flipped 180 degrees at first (i.e.: double curve to the right). Looking at the training set (shown below), it appears as if the only included samples have a "double curve to the left". I assumed that flipping the image by 180 degrees would then result in a correct classification; however, the classifier still failed, classifying the sign as "children crossing". It appears thie is because the sign I selected is a slight variant of the sign in the training set. Note that the signs in the training set have a shorter middle siection, such that the two vertical sections of the "double curve" are closer together. It's also apparent that the training set contained a rather small number of variations of the sign in quetion (i.e.: the samples were derived from a small number of "passes" of the "double curve" sign; I assume that a more varied training set may have improved the outcome. This result is interesting in that it demonstrates potential issues with NNs, and the importance of having a varied training set.

![alt text][image9]


**General Caution**
The classifier correctly predicted this sign, with a probability approaching 100%


**Wild Animals Crossing**
The classifier correctly predicted this sign, with a probability approaching 100%


**Priority Road**
The classifier correctly predicted this sign, with a probability approaching 100%


**Roundabout Mandatory**
The classifier correctly predicted this sign, with a probability of 99.9999%. Interestingly, despite the fac that several other round sights were in the data set, the next closest prediction was a stop sign.


**No Passing**
The classifier correctly predicted this sign, with a probability approaching 100%


[//]: # (Image References)

[image1]: ./md_imgs/hist1.png "Hist1"
[image2]: ./md_imgs/raw_imgs.png "Raw Images"
[image3]: ./md_imgs/unwarped.png "Unwarped"
[image4]: ./md_imgs/warp_example.png "Image Supplementation"
[image5]: ./md_imgs/hist2.png "Hist2"
[image6]: ./md_imgs/proc_imgs.png "Processed Images"
[image7]: ./md_imgs/test_imgs.png "Test Images"
[image8]: ./md_imgs/top5.png "Top 5"
[image9]: ./md_imgs/double_curve.png "Double Curve"
[image10]: ./md_imgs/learning_curve.png "Double Curve"
