# Deep Learning Papers

The papers are listed from newer to older based on the first publication date.

## 2017


### [Visualizing the Loss Landscape of Neural Nets (Hao et al.)](https://arxiv.org/pdf/1712.09913.pdf)

read: ✅ | last edition: 2018
  
The authors visually explain positive effect of skip-connections and wider layers 
in ResNets, as well as properly selected parameters. The is also [a repository](https://github.com/tomgoldstein/loss-landscape) 
with implementation of author's approach. (CLI tool visualizing pre-trained VGG, ResNet and DenseNet 
architectures).

> **Note:** The tool seems to be not too flexible. It is not a library or package that one could 
easily install and drop an instance of pre-trained model into visualizing method. However, it 
should possible to use models different from the models provided by authors.

![](./assets/loss_surface.png)
 
### [Exploring Loss Function Topology with Cyclical Learning Rates (Smith L. N., Topin N.)](https://arxiv.org/pdf/1702.04283.pdf)

read: ✅ | last edition: 2017

The authors discovered a _superconvergence phenomena_ while training residual networks. 
A short paper of 4 pages in total.

> **Quote:** We coin the term **“super-convergence”** to refer to this phenomenon where a network is 
trained to a better final test accuracy compared to traditional training, but with fewer iterations 
and a much larger learning rate.

## 2016

### [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf)

read: ⏳ | last edition: 2016

Using VGG16 model as a loss function on top of image-to-image ResNet model to build with a higher
perceptual attractiveness then when using pixes-wise distance measure.

![](./assets/perceptual_loss.png)

### [A guide to convolution arithmetic for deep learning (Dumoulin V., Visin F.)](https://arxiv.org/pdf/1603.07285.pdf)

read: ✅ | last edition: 2018

A comprehensive guide on convolutions arithmetic. Includes lots of formulas computing number of 
output feature maps depending on input size, kernel size, stride, and padding. The guide helps 
to understand how to pick parameters to reduce/keep/increase the size of output for a specific 
(de)convolution/pooling layer.

The authors include [the link to the repository](https://github.com/vdumoulin/conv_arithmetic) with 
scripts to generate schematic animations of (de)convolution operations.  

![](./assets/no_padding_no_strides.gif)

### 

## 2015

### [Cyclical learning rates for training neural networks (Smith L. N.)](https://arxiv.org/pdf/1702.04283.pdf)

read: ✅ | last edition: 2017

An empirical observation of Circular Learning Rates (CLR) effectiveness on CIFAR-10 and CIFAR-100 datasets. 
Using triangular and exponential learning rate schedulers to improve training convergence speed. 
The author also talks about estimating a good value for the cycle length and `lr_min`/`lr_max` 
boundaries with "LR range test"; run your model for several epochs while letting the learning 
rate increase linearly between low and high LR values.