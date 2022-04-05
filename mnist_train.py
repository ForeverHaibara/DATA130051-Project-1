'''
Author: Zehao Zhang 20307130201
Homepage: https://github.com/ForeverHaibara/DATA130051-Project-1 

The MNIST dataset is available at 
https://academictorrents.com/details/323a0048d87ca79b68f12a6350a57776b6a3b7fb 

A trained model is available at 
https://pan.baidu.com/s/1G9xUypIUgDcwl42_x8xdxw 
with extracting password 'owor'
'''

# data preparation, set the path to your MNIST dataset downloaded from above
path = 'mnist.pkl.gz'


import numpy as np
from neural_network import Network # my Network class
from plotter import plotter        # self-designed plotter
import pickle
import gzip
import time

# data loader
time_start = time.time()
def load_data(path):
    f = gzip.open(path, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)


# training_data  : (50000 * 784, 50000)
# validation_data: (10000 * 784, 10000)
# test_data      : (10000 * 784, 10000)
training_data , validation_data , test_data = load_data(path)
data_x = training_data[0]
data_y = np.eye(10)[training_data[1]]    # onehot encoding
valid_x = validation_data[0]
valid_y = np.eye(10)[validation_data[1]] # onehot encoding


# set up a neural network
np.random.seed(2023)
nn = Network(hidden_size = [784,500,10], acts = ['relu','sigmoid'], lr = 3e-2,
                regws = None, regbs = None)

# if you want to load your model, use the code here
# nn = Network.load(r'mymodel.txt')


# train the model
result = nn.fit(data_x, data_y, epochs = 20, batch_size = 40, loss_func = 'MSE', 
                valid_x = valid_x, valid_y = valid_y, valid_freq = 1)


# read the training result
# losses = result['loss']
# accs   = result['acc'] 


# save your model here
nn.save(r'mymodel.txt')


# check the accuracy on training, validation and testing data
print('-'*30 + '\nAccuracy on Training Data')
nn.predict(data_x, data_y, batch_size = 40, verbose = True)

print('-'*30 + '\nAccuracy on Validation Data')
nn.predict(valid_x, valid_y)

print('-'*30 + '\nAccuracy on Testing Data')
nn.predict(test_data[0], np.eye(10)[test_data[1]])
print(end = '')


print(f'Finished within {time.time() - time_start} seconds')


# plot the loss and accuracy curves
plotter(result)

