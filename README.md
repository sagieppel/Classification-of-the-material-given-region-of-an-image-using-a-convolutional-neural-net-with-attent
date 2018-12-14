# Classification of materials in a given region of the image using a convolutional neural net with attention mask as input

Given an image and region of interest (ROI) mask as input, the net classifies the region of the image marked in the input ROI mask. 

![](/Figure1.png)
Figure 1. Segment-specific Classification using CNN

For example, in Figure 1  the net is given the same image twice with different ROI masks and output the material class of the region marked by each mask.
Weights file of model trained on the 44 materialscategories of the OpenSurfaces materials dataset is supplied.

This net achive 82% classification accuracy on the on the opensurface material data set.

For more details see [Classifying a specific image region using convolutional nets with an ROI mask as input](https://arxiv.org/pdf/1812.00291.pdf)

![](/Figure2.png)
Figure 2.a Standard Classification net, b. Region specific classification net

## Architecture and attention mechanism.
The net architecture can be seen in Figure 2.b. The net main branch consists of standard image classification neural net (Resnet 50). 
The side branch responsible for focusing the attention of the classifier net on the input mask region.
As shown in Figure 2.b attention focusing is done by using the binary ROI mask to generate attention mask wich is the merged by elementwise addition with the feature map of the first layer.

# Using the net.
## Setup
This network was run with [Python 3.6 Anaconda](https://www.anaconda.com/download/) package and [Pytorch](https://pytorch.org/). 

. 

## Prediction/Inference

1. Train net or download pre trained net weight from [here](https://drive.google.com/file/d/1GI_uqwFWUJGr7-g04UufQsIsxmiBSoCv/view?usp=sharing).
2. Open RunPrediction.py 
3. Set path for image in ImageFile
4. Set Path for ROI mask in ROIMaskFile

## Training In train.py
1. Set folder of train images in Image_Dir
2. Set folder for ground truth Annotation in AnnotationDir
    The Label Maps should be saved as png image with same name as the corresponding image and png ending. The value of each pixel correspond to it class
3. Set number of classes number in NUM_CLASSES

## Evaluating 
1. Train net or Download pre trained net weight from [here](https://drive.google.com/file/d/1GI_uqwFWUJGr7-g04UufQsIsxmiBSoCv/view?usp=sharing).
2. Open EvaluateAccuracy.py
3. Set folder of images in Image_Dir
4. Set folder for ground truth Annotation in AnnotationDir
    The Label Maps should be saved as png image with same name as the corresponding image and png ending. The value of each pixel correspond to it class
5. Set number of classes number in NUM_CLASSES
6. Set path to trained model weights in Trained_model_path


