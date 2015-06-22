'''
Package for creating simple neural networks

TODO:
Add regularization parameter
Add ability to choose activation function
Add more activation functions
Add gradients for activation functions
Add advanced optimization using fmincg
Give choice for using advanced optimization or SGD
Should work for any amount of output layers
Stop training when error not changing
Handle overflow and log(0) numerical stuff

'''


import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class Neural_Network():
    
    def __init__(self, layer_sizes, random_seed=None, learning_rate=1):
        '''
        Initialize Neural network

        Parameters
        ----------
        layer_sizes: (Nx1) array or list
            Gives the total size of the NN and the number of Neurons in each layer
            For example: [4,6,6,1] is a NN with an input layer of 4 units, two hidden
            layers with 6 units each and an output layer with 1 unit

        learning_rate: float
            When using SGD, determines the step size

        Notes
        -----

        Randomizes weights for each layer
        '''
        # List of weights for each layer of the network
        self.weights = []
        # List of the cost of the network at each step
        self.cost = [1]
        # DELTA is the computed gradient for each layer. It directly changes the weights
        self.__DELTAS = []
        # error of each node
        self.__deltas = range(len(layer_sizes) - 1)
        # list of activation values for each layer
        self.__a = range(len(layer_sizes))
        # initial values of the weights will be between -epsilon and + epsilon
        self.epsilon = .15
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.__stoping_threshold = 1e-13
        
        #Can set to repeat results
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # initialize weights and set gradients to 0    
        for i, layer_size in enumerate(layer_sizes[:-1]):
            self.weights.append(self.__get_random_weights(layer_sizes[i:i+2]))
            self.__DELTAS.append(np.zeros((layer_size + 1, layer_sizes[i+1])))
                                
    
    def __get_random_weights(self, x):
        '''
        Sets random values for the weights for all layers between +/- epsilon

        Parameters
        ----------
        x: list of 2 values which are the dimensions of the layer desired to get 
        initial weights

        Returns
        -------
        random weights of size x
        '''

        return np.random.rand(x[0] + 1, x[1]) * 2 * self.epsilon - self.epsilon
        
    def __sigmoid(self, x):
        '''
        Applies sigmoid function to all inputted values

        Parameters
        ----------
        x: 2d array of pre-activation values

        Returns
        -------
        Uses sigmoid avtivation function and returns the sigmoid of all pre-activation values
        '''

        return 1 / (1 + np.exp(-x))

    def __tanh(self, x):
        '''
        Applies hyperbolic tangent function to all inputted values

        Parameters
        ----------
        x: 2d array of pre-activation values

        Returns
        -------
        Uses tanh avtivation function and returns the tanh of all pre-activation values
        '''
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def __sigmoid_gradient(self, x):
        '''
        Finds the derivative of the sigmoid function

        Parameters
        ----------
        x: 2d array

        Returns
        -------
        Gradient of x
        '''
        return np.multiply(x, 1-x)
    
    def feed_forward(self, X):
        '''
        Feeds input through one iteration of the neural network to get thee output

        Parameters
        ----------
        X: 2d array of inputs

        Returns:
        --------
        Output of NN
        '''

        # multiply weights times input, then activate and repeat
        for i, theta in enumerate(self.weights):
            z = np.dot(self.__a[i], theta)
            self.__a[i+1] = np.c_[np.ones(len(z)), self.__sigmoid(z)]
        # Remove column of ones in last layer  
        self.__a[-1] = self.__a[-1][:,1:]
    
    def back_prop(self, y):
        '''
        Backpropagation algorithm. Attempts to find "error" in each layer and use SGD
        to move weights closer to optimum

        Parameters
        ----------
        y: given final output

        Returns
        -------
        None

        Notes
        -----
        Sets weights of NN object
        Finds errors by going in reverse order from output to input
        '''

        #Gets error in last layer first
        # Do not need to multiply by sigmoid gradient if using MLE as cost function
        # for better explanation http://neuralnetworksanddeeplearning.com/chap2.html
        self.__deltas[-1] = np.multiply(self.__a[-1] - y, self.__sigmoid_gradient(self.__a[-1]))

        for i in range(-1, -len(self.__deltas), -1):
            self.__deltas[i-1] = np.multiply(np.dot(self.__deltas[i], self.weights[i].T), self.__sigmoid_gradient(self.__a[i-1]))[:, 1:]

        for i, D in enumerate(self.__DELTAS):
            self.__DELTAS[i] = np.dot(self.__a[i].T, self.__deltas[i])
            self.weights[i] -= self.learning_rate * self.__DELTAS[i] / len(y)
    
    def cost_function(self, y, h, m):
        '''
        MLE cost function

        Parameters
        ----------
        y: Given target values
        h: Output of feed forward
        m: Number of examples

        Returns
        -------
        Negative log-likelihood
        '''
        return -1./m * np.sum(y * np.log(h) + (1-y) * np.log(1-h))

    def cost_function_se(self, y, h, m):
        '''
        Calculates the squared error of the cost function

        Parameters
        ----------
        y: Given target values
        h: Output of feed forward
        m: Number of examples

        Returns
        -------
        Squared error divided by two
        '''

        return 1./m * np.sum((y - h) ** 2) / 2
        
    def predict(self, X):
        '''
        Feed forward to predict

        Parameters
        ----------
        X: 2d array of inputs

        Returns
        -------
        Predicted output
        '''
        a = np.c_[np.ones(len(X)), X]
        for i, theta in enumerate(self.weights):
            z = np.dot(a, theta)
            a = np.c_[np.ones(len(z)), self.__sigmoid(z)]
            
        return a[:,1:]
        
        
    def fit(self, X, y):
        '''
        Attempts to find best parameters

        Parameters
        ----------
        X: 2d array of inputs
        y: 2d array of targets

        Returns
        -------
        None

        Notes
        -----
        Can now predict
        '''

        self.__a[0] = np.c_[np.ones(len(X)), X]
    
        for i in range(50000):
            self.feed_forward(X)
            
            current_cost = self.cost_function_se(y, self.__a[-1], len(y))
            self.cost.append(current_cost)
            if abs(self.cost[-1] - self.cost[-2]) < self.__stoping_threshold:
                print "cost not decreasing", i
                break
    
            self.back_prop(y)

    def score(self, X, y):
        '''
        Returns Accuracy of prediction

        Parameters
        ----------
        X: 2d array of inputs
        y: 2d array of outputs

        Returns
        -------
        Accuracy (float)
        '''
        return np.mean(np.round(self.predict(X)) == y)

if __name__ == '__main__':
    # # xor exmaple
    # X = np.array([[0,0], [0,1], [1,0], [1,1]])
    # y = np.array([[0,1,1,0]]).T
    # nn = Neural_Network([2,2,1], learning_rate=1)
    # nn.fit(X, y)
    # print nn.predict(X)

    X = np.random.rand(300,2)
    y = np.atleast_2d(1 * np.logical_or((X[:,0] - .8) ** 2 + (X[:,1]-.5) ** 2 < .03, (X[:,0] - .2) ** 2 + (X[:,1]-.8) ** 2 < .03)).T

    nn = Neural_Network([2,10,1], learning_rate=5)
    nn.fit(X,y)

    xx, yy = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
    Z = np.round(nn.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,8))
    plt.contourf(xx, yy, Z, alpha=.8)
    plt.scatter(X[:,0], X[:,1], c=y, s=60)
    plt.title("Neural Net Accuracy is " + str(np.round(nn.score(X,y),2)))
    plt.show()
    
