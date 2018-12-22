# Deep Learning Papers

The papers are listed from newer to older based on the first publication date.

## 2017


### 1. [Visualizing the Loss Landscape of Neural Nets (Hao et al.)](https://arxiv.org/pdf/1712.09913.pdf)

Read: ✅
Last edition: 2018
  
The authors visually explain positive effect of skip-connections and wider layers 
in ResNets, as well as properly selected parameters. The is also [a repository](https://github.com/tomgoldstein/loss-landscape) 
with implementation of author's approach. (CLI tool visualizing pre-trained VGG, ResNet and DenseNet 
architectures).

> **Note:** The tool seems to be not too flexible. It is not a library or package that one could 
easily install and drop an instance of pre-trained model into visualizing method. However, it 
should possible to use models different from the models provided by authors.

![](./assets/loss_surface.png)
 

## 2016

### 2. [A guide to convolution arithmetic for deep learning (Dumoulin V., Visin F.)](https://arxiv.org/pdf/1603.07285.pdf)

Read: ✅
Last edition: 2018

A comprehensive guide on convolutions arithmetic. Includes lots of formulas computing number of 
output feature maps depending on input size, kernel size, stride, and padding. The guide helps 
to understand how to pick parameters to reduce/keep/increase the size of output for a specific 
(de)convolution/pooling layer.

The authors include [the link to the repository](https://github.com/vdumoulin/conv_arithmetic) with 
scripts to generate schematic animations of (de)convolution operations.  

![](./assets/no_padding_no_strides.gif)