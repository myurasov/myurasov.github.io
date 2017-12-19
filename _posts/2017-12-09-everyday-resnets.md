---
layout: post
title:  "Everyday ResNets"
date:   2017-12-09 19:47:00 -0800
comments: true
thumb: "assets/posts/ssd-tx1/d.jpg"
---

ResNets for everyday use.

- what are ResNets
- why you need one (vs vgg-like cnns, vs pretrained models)
- constructing resnet18
- options: avg pooling, more fc layers, adding dropouts


CNNs are widely used in many Deep Learning applications, not only in image-related ones, but also in speech recognition, EEG/ECG analysis, etc. – any task where pattern recognition is needed. When it comes to achieving practical results quickly, best thing is often to try to adapt your approach and data to use a high-accuracy pre-trained model. For times when it is feasible to train the entire model from scratch and some CNN is needed, VGG-like architectures are often used. This is likely due to the straightforward implementation, stability in training and the fact that feature vectors from the top layers can serve as relevant input for transfer learning tasks or other branches of the model. However on many tasks ResNets have shown advantages over VGG-like nets such as achieving better accuracy with less computational power needed, better generalization and lesser risk of picking too large model for a task (more on this later). This post gives a few tips on how to implement a ResNet-like architecture for your everyday Deep Learning needs.

# What is ResNet?

The idea behind ResNet is to ensure a deeper network (hence with more expressive power) is actually capable of learning what a shallower network can. When more layers are simply stacked together, number of problems emerge: vanishing/exploding gradients, slow convergence and degradation that is not caused by overfitting and leads to error growth with extra layers added.

Authors of ResNet suggested that in may be useful to help network to learn an identity function in certain layers, effectively skipping them as needed. To do that they introduced _identity connections_ – simple addition of output of a deeper layer with a current one:

<center>
<img src="{{ site.url }}/assets/posts/resnets/resnet-a.png" alt="Identity connections in ResNet" width="40%">
</center>

This addition not only allows network to easily learn anything a shallower one can by effectively skipping certain convolutional layers, but also propagate gradients with less modification and therefore much deeper.

# Constructing ResNet 

Authors of ResNet proposed 18, 34, 50, 101 and 152-layer versions of the network. They all share the same structure, the difference is purely in number and size of blocks used between base convolutional layer and top 1000-d densely connected layer. Here is a general template for constructing one:

- Input layer
- Base convolutional block (64 7x7 filters, stride)


# Links

- Deep Residual Learning for Image Recognition (original ResNet paper) by He, Zhang, Ren, Sun – [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf)
# ++
# @see https://github.com/raghakot/keras-resnet/blob/master/resnet.py
# @see https://gist.github.com/JefferyRPrice/c1ecc3d67068c8d9b3120475baba1d7e
# @see https://wiseodd.github.io/techblog/2016/10/13s/residual-net/
# @see https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
# @see http://torch.ch/blog/2016/02/04/resnets.html