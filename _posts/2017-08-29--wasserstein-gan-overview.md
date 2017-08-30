---
layout: post
title:  "Wasserstein GAN – Quick Overview"
date:   2017-08-29 17:30:00 -0800
comments: true
thumb: "assets/posts/ssd-tx1/d.jpg"
---
<!-- 
- brief history and overview of generative models, gen models applications
- why gans and gen models are useful
- tree from Goddfellow's tutorial
- gans training explaned + drawing/schema
- what are wasserstein gans ('D is a cost function that estimates EM distance between trues (priors) and generated'), what problems they attempt to solve
- Implementing WGANS in keras - code walk-through + some pics
- conclusion (wgans vs gans, other novell GAN types)
- links

- split into 2 parts (GANS, Implementing WGAN)?


""This post comes from my experience trying to replicate --paper on WGANs--""

-- preface -- -->



<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [GAN?](#gan)
- [Wasserstein GAN?](#wasserstein-gan)
- [Code!](#code)
- [Links](#links)

<!-- /TOC -->

While reading the [Wasserstein GAN paper](https://arxiv.org/pdf/1701.07875.pdf) I decided that the best way to understand it is to code it. This is a quick overview of the paper itself and is followed by the actual [code]() in Keras.


# GAN?

GAN, or Generative Adversarial Network is a type of _generative model_ – a model that looks at the training data drawn from a certain distribution and tries to estimate that distribution. New samples obtained from such model "look" alike original training samples. Some generative models only learn the parameters of training data distribution, some only able to draw samples from it, some allow both.

Many types of generative models exist: Fully Visible Belief Networks, Variational Autoencoders, Boltzmann Machine, Generative Stochastic Networks, PixelRNNs, etc. They differ in a way they represent or approximate the density of the training data. Some construct it explicitly, some just provide ways to interact with it – like generating samples. GANs fall into latter category. Most generative models learn by maximum likelihood estimation principle – by choosing the parameters of the model in such a way that it assigns maximum probability ("likelihood") to training data.   

GANs work by playing a game between two components: Generator (_G_) and Discriminator (_D_) (both usually represented by neural nets). Generator takes a random noise as input and tries to produce samples in such a way that Discriminator is unable to determine if they come from training data or Generator. Discriminator learns in a supervised manner by looking at real and generated samples and labels telling where those samples came from. In some sense, Discriminator component replaces the fixed loss function and tries to learn one that is relevant for the distribution from which the training data comes.

# Wasserstein GAN?

In essence the goal of any generative model is to minimize the difference (divergence) between real data and model (learned) distributions. However in traditional GAN _D_ doesn't provide enough information to estimate this distance when real and model distributions do not overlap enough – which leads to weak signal for _G_ (especially in the beginning of the training) and general instability.

Wasserstein GAN adds few tricks to allow _D_ to approximate [Wasserstein (aka Earth Mover's) distance](https://en.wikipedia.org/wiki/Wasserstein_metric) between real and model distributions. Wasserstein distance roughly tells "how much work is needed to be done for one distribution to be adjusted to match another" and is remarkable in a way that it is defined even for non-overlapping distributions.

For _D_ to effectively approximate Wasserstein distance:
- It's weights have to lie in a [compact space](https://en.wikipedia.org/wiki/Compact_space). To enforce this they are clipped to a fixed box ([-0.01, 0.01]) after each training step. However authors admit that this is not ideal and highly sensitive to clipping interval chosen, but works in practice. See p. 6-7 of the [paper](https://arxiv.org/pdf/1701.07875.pdf) for more on this.
- It is trained to much more optimal state so it can provide _G_ with a useful gradients.
- It should have linear activation on top.
- It is used with a cost function that essentially doesn't modify it's output

  ```python
  K.mean(y_true * y_pred)
  ```

    where:
    
    - mean is taken to accommodate different batch sizes and multiplication 
    - predicted values are multiplied element-wise with true values which can take -1 to allow output to be maximized (optimizer always tries to minimize loss function value)

Authors claim that compared to vanilla GAN, WGAN has the following benefits:

- Meaningful loss metric. _D_ loss correlates well with quality of generated samples which allows for less monitoring of the training process.
- Improved stability. When the _D_ is trained till optimality it provides a useful loss for _G_ training. This means training of _D_ and _G_ doesn't have to be balanced in number of samples (it has to be balanced in vanilla GAN approach). Also authors claim that in no experiments they experienced a mode collapse happening with WGAN.

# Code!

Let's dust off the keyboard and implement [Wasserstein GAN in Keras ⇢]().

# Links

1. [Wasserstein GAN paper](https://arxiv.org/pdf/1701.07875.pdf) – Martin Arjovsky, Soumith Chintala, Léon Bottou
1. [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf) – Ian Goodfellow
1. [Original PyTorch code for the Wasserstein GAN paper](https://github.com/martinarjovsky/WassersteinGAN)
