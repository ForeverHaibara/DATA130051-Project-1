<h1 align="center"> DATA130051 Project 1</h1>

<div align="center"> Repository for Course DATA130051 Project 1</div>

<div align="center"> Author: 20307130201 张泽豪</div>

## Contents

- [Assignment](#assignment)
  * [Training](#training)
  * [Hyperparameter Tuning](#hyperparameter-tuning)
  * [Testing](#testing)
- [Requirements](#basic-requirements)
- [Instructions](#instructions)


## Assignment
Construct a two-layer neural network classifier including the three parts below. 

### Training
* Activation Functions
* Back Propagation with Loss and Gradient Computation
* Learning Rate Decay
* L2-Regularization
* Optimizer SGD
* Model Saving

### Hyperparameter Tuning 
* Learning Rate
* Hidden Layer Size
* Regularization

### Testing
Load the model with tuned hyperparameters and output the 
classification accuracy. 

<br>

Utilize the MNIST dataset, see more info at http://yann.lecun.com/exdb/mnist/. 

**DO NOT** use PyTorch, TensorFlow or any deep-learning python packages. NumPy is allowed. 

Upload your code onto your github repository with instructions on training and testing process in a README file. The trained model should be uploaded onto a cloud drive. 

## Basic Requirements

1. Python 3
2. Numpy
3. Tqdm


## Instructions 

* **neural_network.py**

This is where we implement the Network class. One can customize arbitrary DNN models with it. 

* **Dataset**

Download the MNIST dataset in pkl.gz format 
<a href= https://academictorrents.com/details/323a0048d87ca79b68f12a6350a57776b6a3b7fb>here</a>. It is split into training set, validation set and the testing set with 50000, 10000 and 10000 figures respectively. 

* **Model**

A trained model with 98.46% accuracy on testing data is available at the <a href = https://pan.baidu.com/s/1G9xUypIUgDcwl42_x8xdxw>cloud drive</a>
with extracting password '**owor**'. Use the following code to load the model.

`nn = Network.load('mymodel.txt')`

* **plotter.py**

An auxiliary function for plotting the training history. It requires Matplotlib.

* **mnist_train.py**

It illustrates an example of training the model and plotting the loss functions. With the given random seed anyone is able to reach a 99.994% accuracy on training data and 98.46% accuracy on testing data. It also involves instructions on loading and saving the model.

* **mnist_train.ipynb**

In the Jupyter notebook includes hyperparameter searching and visualizations besides simply training, saving and loading models. The last step, visualizing the weights by PCA, requires the python package Sklearn.
