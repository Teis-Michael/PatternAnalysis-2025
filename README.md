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

# working principles 
algorithm implemented, problem solved. 
the dataset consists of a image and a segmentation image that acts as the image mask.
these are used in a Unet model that consists of _x_ layers of _y_ including a encoder, decoder.
this is then trained with epochs and learning rate of _z_.

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
inputs
# output
outputs

# justify
describe, pre-processing, ref, justify train, valid, test
