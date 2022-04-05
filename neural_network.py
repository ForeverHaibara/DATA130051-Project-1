'''
Author: Zehao Zhang 20307130201
Homepage: https://github.com/ForeverHaibara/DATA130051-Project-1 

A dense neural network class implementation.
'''

import numpy as np
from tqdm import tqdm

class Activator():
    '''Vectorized activators and its derivative'''
    def __init__(self):
        pass

    @classmethod
    def derivative_sigmoid(self, x):
        t = 1. / (1 + np.exp(-np.clip(x, -30, 30)))
        return t * (1. - t)

    @classmethod
    def activate(self, name = 'relu'):
        if   name == 'relu':    return lambda x: np.where(x > 0 , x , 0)
        elif name == 'sigmoid': 
            return lambda x: 1. / (1 + np.exp(-np.clip(x, -30, 30)))

    @classmethod
    def derivative(self, name = 'relu'):
        if   name == 'relu':    return lambda x: np.where(x > 0 , 1 , 0)
        elif name == 'sigmoid': return Activator.derivative_sigmoid

    
class Network():
    '''
    A simple dense neural network (DNN) implementation using NumPy.

    Support arbitrary number of DNN layers with activations through.
    '''
    def __init__(self, 
            hidden_size = [784,100,10],  # size of data in hidden layers
            acts = ['relu','sigmoid'],   # activator typename 
            regws = None,                # l2-regularization penalty on weights
            regbs = None,                # l2-regularization penalty on biases
            lr    = 1e-2                 # learning rate
            ):
        '''
        Initialize the model with given hyperparameters.

        Parameters
        ----------
        hidden_size: array-like, length (L+1)
            Size of data in hidden layers.

        acts: list of str, length L
            Activator typenames. Both 'relu' and 'sigmoid' are supported.

        regws: array-like, length L
            L2-regularizations on weights. Defaults to none.

        regbs: array-like, length L
            L2-regularizations on biases. Defaults to none.

        lr: float
            Initial learning rate.
        '''

        self.hidden_size = hidden_size
        self.weights = []
        self.biases = []
        self.acts = acts
    
        # random initialization
        for w, h in zip(hidden_size[:-1], hidden_size[1:]):
            self.weights.append( np.random.randn(w, h) * .1 )
            self.biases .append( np.random.randn(1, h) * .1 )

            self.weights[-1] = self.weights[-1].astype('float32')
            self.biases[-1]  = self.biases[-1] .astype('float32')
        
        self.z     = [None for _ in range(len(self.weights)+1)]  # input of each layer
        self.grads = [None for _ in range(len(self.weights)+1)]  # grads for weights
        self.regws = [0] * len(self.weights) if regws is None else regws
        self.regbs = [0] * len(self.weights) if regbs is None else regbs
        self.lr    = lr
        
    def forward(self, x):
        '''compute model(x) and automatically save the inputs for backprop'''
        self.z[0] = x
        for i in range(len(self.acts)):
            # Dense Layer
            # store the forward x in z[i+1]
            self.z[i+1] = self.z[i] @ self.weights[i] + self.biases[i]

            # Activator
            self.z[i+1] = Activator.activate(self.acts[i])(self.z[i+1])
        return self.z[-1]
    
    def __call__(self, x):
        return self.forward(x)

    def backprop(self, dx):
        '''backprop the gradients'''
        for i in range(len(self.acts)-1, -1, -1):
            # Activator Derivative (pointwise multiplication)
            dx = Activator.derivative(self.acts[i])(self.z[i+1]) * dx

            # Dense Layer Derivative
            self.grads[i] = self.z[i].T @ dx

            # Back propagation
            dx = dx @ self.weights[i].T

    def update(self, lr):
        '''update the parameters by SGD'''
        for i in range(len(self.weights)):
            if self.grads[i] is not None:
                if self.regws[i] != 0:
                    self.weights[i] -= self.weights[i] * self.regws[i]
                if self.regbs[i] != 0:
                    self.biases [i] -= self.biases [i] * self.regbs[i]
                self.weights[i] -= self.grads[i] * lr

    def compute_loss(self, pred, fact, batch_size, loss_func = 0):
        # compute loss
        v = 0
        if loss_func == 0:   # MSE
            v = .5 * np.sum(np.square(pred - fact)) / batch_size
        elif loss_func == 1: # BCE
            v = -1. / batch_size * np.sum(
                fact*np.log(pred) + (1 - fact)*np.log((1+2e-7) - pred))
            # Here we add 2e-7 to prevent log(0)
            
        # add regularization loss
        for i in range(len(self.weights)):
            if self.regws[i] != 0:
                v += .5 * self.regws[i] * np.sum(np.square(self.weights[i]))
            if self.regbs[i] != 0:
                v += .5 * self.regbs[i] * np.sum(np.square(self.biases [i]))

        return v

    def fit(self, x, y, epochs = 1, batch_size = 40, loss_func = 'MSE',
            valid_x = None, valid_y = None, valid_freq = 1,
            pre_result = None, accs = None, lr = None):
        '''
        Fit the model to dataset (x,y).

        Parameters
        ----------
        x: array-like, shape (N,X)
            Input data.

        y: array-like, shape (N,Y)
            Labels. Should be onehot encoded for classifcations. 

        epochs: int
            Number of epochs trained on the dataset.

        batch_size: int
            Size per batch on SGD.

        loss_func: str, 'MSE' or 'BCE'
            Loss function, standing for mean-squared-loss or binary-cross-entropy.
        
        valid_x: array-like, shape (M,X)
            Validation data. Defaults to None to disable validation

        valid_y: array-like, shape (M,Y)
            Validation labels. Defaults to None.
        
        valid_freq: int
            Validate on the validation data every certain epochs. 

        pre_result: dict
            The training will append the result in the dictionary. Defaultedly None.

        lr: float
            Learning rate. Defaults to the current learning rate of the model.         

        Returns
        ----------
        Returns a dictionary with two keys, 'loss' and 'acc'.

        loss: list
            The list of losses during training.

        acc: list
            The list of validation accuracies.
        '''

        # defaults
        if lr is not None: self.lr = lr
        loss_func = {'MSE': 0, 'BCE': 1}[loss_func]

        # inherit the results
        if pre_result is not None:
            losses = pre_result['loss']
            accs   = pre_result['acc']
            losses_valid = pre_result['loss_valid']
        else:
            losses = []
            accs   = []
            losses_valid = []
        

        n = x.shape[0] 
        for epoch in range(1, 1 + epochs):
            # random shuffled indices
            shuffle = np.arange(n)
            np.random.shuffle(shuffle)

            batch_e = 0
            for _ in tqdm(range(n // batch_size)):
                # extract a batch
                batch_s = batch_e 
                batch_e = batch_s + batch_size
                batch_x = x[shuffle[batch_s: batch_e]] 
                batch_y = y[shuffle[batch_s: batch_e]]

                # forward pass
                predict_y = self.forward(batch_x)

                # compute loss
                losses.append( self.compute_loss(predict_y, batch_y, batch_size, loss_func) )

                # back propagation
                if loss_func == 0:   # MSE
                    self.backprop(predict_y - batch_y)
                elif loss_func == 1: # BCE
                    self.backprop( 
                        np.true_divide(predict_y - batch_y, predict_y * ((1+1e-5) - predict_y)))
                    # Here we add 1e-5 to prevent division by zero or overflow

                # update
                self.update(self.lr)

            if valid_x is not None and epoch % valid_freq == 0:
                # validate
                valid_result = self.predict(valid_x, valid_y, batch_size, 
                                            loss_func = loss_func, verbose = False)
                accs.append(valid_result[0])
                losses_valid.append(valid_result[1])

                # learning rate decay
                if len(accs) > 1 and accs[-1] < accs[-2]:
                    self.lr *= .5


        return {'loss': losses, 'acc': accs, 'loss_valid': losses_valid}

    def predict(self, x, y = None, batch_size = 40, loss_func = None, verbose = True):
        '''
        Make prediction on x or 
        validate the classification accuracy on (x,y) (if y is given).
        '''
        if y is not None: # predict with validation
            batch_e = 0
            n = x.shape[0]
            acc , loss = 0 , 0

            iterator = range(n // batch_size)
            if verbose: iterator = tqdm(iterator)
            for _ in iterator:
                # extract a batch
                batch_s = batch_e 
                batch_e = batch_s + batch_size
                batch_x = x[batch_s: batch_e]
                batch_y = y[batch_s: batch_e]

                # forward pass
                predict_y = self.forward(batch_x)

                acc += np.sum(batch_y[np.arange(batch_size),
                                    np.argmax(predict_y, axis = -1)])
                
                loss += self.compute_loss(predict_y, batch_y, batch_size, loss_func)
            
            loss /= (n // batch_size) # mean loss

            if verbose: print(f'Acc = {int(acc)}/{n} = {acc * 100. / n}%')
            return acc * 1. / n , loss

        else:  # predict with classification
            batch_e = 0
            n = x.shape[0]
            acc , loss = 0 , 0
            predicts = np.zeros(1, dtype='int16')

            iterator = range(n // batch_size)
            if verbose: iterator = tqdm(iterator)
            for _ in iterator:
                # extract a batch
                batch_s = batch_e 
                batch_e = batch_s + batch_size
                batch_x = x[batch_s: batch_e]

                # forward pass
                predict_y = self.forward(batch_x)
                predict_y = np.argmax(predict_y, axis = -1).flatten()
                predicts = np.hstack((predicts, predict_y))
            
            return predicts[1:]

    @classmethod
    def load(self, path):
        '''Load a model from a path and return the model.'''
        def arrayload(x, f):
            # load a 2d array x from buffer f
            n = x.shape[0]
            for i in range(n):
                x[i] = np.array([float(i) for i in f.readline()[:-1].split()])

        with open(path, 'r') as f:
            hidden_size = [int(i) for i in f.readline()[:-1].split()]
            acts = f.readline()[:-1].split()
            model = Network(hidden_size, acts)
            for i in range(len(hidden_size) - 1):
                arrayload(model.weights[i], f)
                arrayload(model.biases [i], f)
        return model

    def save(self, path):
        '''Write (save) a model to the path in .txt format.'''
        def arraysave(x, f):
            # write a 2d array x through buffer f
            for line in x:
                f.write('\n' + ' '.join(['%.6f'%value for value in line]))
        
        with open(path,'w') as f:
            f.write(' '.join([str(i) for i in self.hidden_size]))
            f.write('\n' + ' '.join([str(i) for i in self.acts]))
            for i in range(len(self.weights)):
                arraysave(self.weights[i], f)
                arraysave(self.biases [i], f)


if __name__ == '__main__':
    pass

