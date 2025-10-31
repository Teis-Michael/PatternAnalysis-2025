# Pattern Analysis
Pattern Analysis of various datasets by COMP3710 students in 2025 at the University of Queensland.

We create pattern recognition and image processing library for Tensorflow (TF), PyTorch or JAX.

This library is created and maintained by The University of Queensland [COMP3710](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) students.

The library includes the following implemented in Tensorflow:
* fractals 
* recognition problems

In the recognition folder, you will find many recognition problems solved including:
* segmentation
* classification
* graph neural networks
* StyleGAN
* Stable diffusion
* transformers
etc.

## COMP3710 s4801908 project 1 
Segment the 2D OASIS brain data set with an Improved UNet [1] or 2D CAN [2] with all labels having a minimum Dice similarity coefficient of 0.9 on the test set. [Easy Difficulty]

algorithm: Improved UNet for image segmentsation
problem: segment 2D OASIS brain scans

semantic segmentation 

## archetecture 
algorithm implemented, problem solved. 
the dataset consists of a image and a segmentation image that acts as the image mask.
these are used in a Unet model that consists of 3 convolution blocks for both encoder, decoder.
the Unet takes in a batch_sizex1x256x256 image and outputs batch_size_64x64 probability

# hyper parameter
this is then trained with epochs _z_
and learning rate of _z_.

# dependencies
scikit-learn 1.7.1
matplotlib 3.10.0
numpy 2.3.2
torch 2.8
torchmetric 1.8.
PIL 11.1.0

# reproducity of results
some variation in end results. <ins>random selection from dataset</ins>

## example
# input
from the 2D OASIS brain data and input image and mask where collected. 
'display image of inputs'
''from png labeled slices and seg. ''
the mask is resize to 64x64 using a nearest interpolation method
the image pile is a 256x256 grayscale the was normalise to a mean of 0 and standard deviation of 1
'display mask after processing'
# output
// current 
outputs a //check output size// with 1 channel. this is compared with a dice loss function to determine loss.  
'display output'
'display loss over epoch'
the outputs have a degrees of randomness. this is particalue due to some randomness in the 'adam optimiser' and datasetloader

# references

https://colab.research.google.com/drive/1VOsZSyRhyuHLmgoqGriQk01ub4bKNmZ1?usp=sharing#scrollTo=23402ec0
https://github.com/shakes76/PatternAnalysis-2024/pull/90
https://github.com/shakes76/PatternAnalysis-2024/pull/113
https://github.com/shakes76/PatternAnalysis-2024/pull/178
https://github.com/shakes76/PatternAnalysis-2024/pull/138
