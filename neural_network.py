'''
Package for creating simple neural networks

TODO:

Add more activation functions
Stop training when error not changing
Handle overflow and log(0) numerical stuff
Add regression functionality

'''
from __future__ import division
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class Neural_Network():
    
    def __init__(self, layer_sizes, random_seed=None, learning_rate=1, opt=False, \
                 epsilon=.15, activation_func='sigmoid', epochs=200000, check_gradients=False, C=1):
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
        self.cost = [0]
        # DELTA is the computed gradient for each layer. It directly changes the weights
        self.__DELTAS = []
        # error of each node
        self.__deltas = range(len(layer_sizes) - 1)
        # list of activation values for each layer
        self.__a = range(len(layer_sizes))
        # initial values of the weights will be between -epsilon and + epsilon
        self.epsilon = epsilon
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.__stoping_threshold = 1e-13
        self.opt = opt
        self.epochs = epochs
        self.check_gradients = check_gradients
        self.C = C
        
        #get correct activation function and activation function derivative
        if activation_func == 'sigmoid':
            self.__activation_func = self.__sigmoid
            self.__activation_gradient = self.__sigmoid_gradient
        elif activation_func == 'tanh':
            self.__activation_func = self.__tanh
            self.__activation_gradient = self.__tanh_gradient
        else:
            raise ValueError(activation_func + " is not recognized as an activation function. " +
                "Please choose 'sigmoid' or 'tanh'")
            
        
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
    
    def __tanh_gradient(self, x):
        '''
        Finds the derivative of the tanh function

        Parameters
        ----------
        x: 2d array

        Returns
        -------
        Gradient of x
        '''
        return 1.0 - x**2
    
    def __feed_forward(self, X):
        '''
        Feeds input through one iteration of the neural network to get the output

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
            self.__a[i+1] = np.c_[np.ones(len(z)), self.__activation_func(z)]
        # Remove column of ones in last layer  
        self.__a[-1] = self.__a[-1][:,1:]
    
    def __back_prop(self, change_weights=True):
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
        
        # Using error function E = (y' - y) ^ 2
        self.__deltas[-1] = np.multiply(self.__a[-1] - self.__y_encoded_mat, self.__activation_gradient(self.__a[-1]))

        for i in range(-1, -len(self.__deltas), -1):
            self.__deltas[i-1] = np.multiply(np.dot(self.__deltas[i], self.weights[i].T), self.__activation_gradient(self.__a[i-1]))[:, 1:]

        for i, D in enumerate(self.__DELTAS):
            regularized_weights = self.weights[i] * self.C
            regularized_weights[:,0] = 0
            self.__DELTAS[i] = (np.dot(self.__a[i].T, self.__deltas[i])  + regularized_weights) / len(self.__y_encoded_mat)
            if change_weights:
                self.weights[i] -= self.learning_rate * self.__DELTAS[i] 
    
    def __convert_weights_back_to_matrix(self, x):
        '''
        Rolls weights back up into matrices
        
        Parameters
        ----------
        x: unrolled parameters
        '''
        cum_sum = [0]
        for i, layer in enumerate(self.layer_sizes[:-2]):

            cum_sum.append((layer + 1) * self.layer_sizes[i + 1] + cum_sum[-1])
            self.weights[i] = x[cum_sum[-2]:cum_sum[-1]].reshape(layer+1, self.layer_sizes[i + 1])
        
        self.weights[-1] = x[cum_sum[-1]:].reshape(self.layer_sizes[i + 1] + 1, self.layer_sizes[i + 2])
    
    
    def __cost_func_opt(self, x, *args):
        '''
        Cost function used for advanced optimization
        
        Parameters
        ----------
        x: unrolled parameters
        '''
        self.__convert_weights_back_to_matrix(x)
        self.__feed_forward(X)
        return self.__cost_function_se()
    
    
    def __gradf(self, x, *args):
        '''
        Gradient function used for advanced optimization
        
        Parameters
        ----------
        x: unrolled parameters
        '''
        self.__convert_weights_back_to_matrix(x)
        self.__feed_forward(X)
        self.__back_prop(change_weights=False)
        
        ALL_DELTAS = self.__DELTAS[0].ravel()
        for D in self.__DELTAS[1:]:
            ALL_DELTAS = np.r_[ALL_DELTAS, D.ravel()]
        return ALL_DELTAS
    

    def __grad_check(self, theta, X):
        '''
        Manually checks the gradient. Prints out the MSE of the gradient found with respect to 
        back propogation and differencing the cost function over a small value of epsilon.
        This should yield a value very small less than 1e-10
        
        Parameters:
        -----------
        theta: unrolled parameters of nn
        
        X: Input data
        '''
        grads = np.zeros(len(theta))
        epsilon = 1e-4
        for i, grad in enumerate(grads):            
            theta_plus = theta.copy()
            theta_plus[i] += epsilon
            self.__convert_weights_back_to_matrix(theta_plus)
            self.__feed_forward(X)
            cost_plus = self.__cost_function_se()
            
            theta_minus = theta.copy()
            theta_minus[i] -= epsilon
            self.__convert_weights_back_to_matrix(theta_minus)
            self.__feed_forward(X)
            cost_minus = self.__cost_function_se()
            
            grads[i] = (cost_plus - cost_minus) / (2*epsilon)
        
        self.__convert_weights_back_to_matrix(theta)
        self.__feed_forward(X)
        self.__back_prop()
        ALL_DELTAS = self.__DELTAS[0].ravel()
        for D in self.__DELTAS[1:]:
            ALL_DELTAS = np.r_[ALL_DELTAS, D.ravel()]
        
        print "manual grad check MSE", np.sum((grads - ALL_DELTAS)**2) / len(grads) 
       
    
    def __cost_function_mle(self, y, h, m):
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

    def __cost_function_se(self):
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
        regularized_sum = 0
        for weight in self.weights:
            regularized_sum += np.sum(weight[:,1:]**2)
        regularized_sum = regularized_sum * self.C / (2*len(self.__y_encoded_mat))
        
        return 1/len(self.__y_encoded_mat) * np.sum((self.__y_encoded_mat - self.__a[-1]) ** 2) / 2 + regularized_sum
        
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
            a = np.c_[np.ones(len(z)), self.__activation_func(z)]
        
        if len(self.classes_) > 2:
            soft_max = np.argmax(a[:,1:], axis=1)
            reverse_lookup = {v:k for k,v in self.__encoded_classes.iteritems()}
            returned_prediction = np.array([reverse_lookup[x] for x in soft_max])
        else:
            returned_prediction = np.round(a[:,1:])
            
        return returned_prediction
    
    def __encode_y_values(self, y):
        '''
        Encodes numeric and string values into a matrix of 0's and 1's
        
        Parameters
        ----------
        y: A 1 dimensional array or 2 dimensional array with 1 column of target values
        
        Returns:
        -------
        None
        
        Notes
        -----
        Creates a mapping of the unique inputs into a 2 dimensional matrix where each row
        represents a y-value. Each row will consist of zeros except for a 1 for the appropriate y value
        '''
        if np.ndim(y) == 2 and y.shape[1] != 1:
            raise ValueError("bad y shape {0}. Make y 1 dimension or have 1 as its second dimension".format(y.shape))
        
        self.classes_ = np.unique(y)
        self.__encoded_classes = {cls:i for i, cls in enumerate(self.classes_)}
        
        if type(y) == np.ndarray:
            y = y.ravel()
        self.__y_encoded = np.atleast_2d(np.array([[self.__encoded_classes[cls] for cls in y]]).T)
        
        self.__y_encoded_mat = self.__y_encoded.copy()
        if len(self.classes_) > 2:
            self.__y_encoded_mat = np.zeros((len(self.__y_encoded), len(self.classes_)))
            for i, y_encoded in enumerate(self.__y_encoded):
                self.__y_encoded_mat[i, y_encoded] = 1
        
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
        if len(X) != len(y):
            raise ValueError('X and y have incompatible shapes. X has ' + str(len(X)) + ' examples but ' +
                             'y has ' + str(len(y)) + '.')
        
        
            
        self.__encode_y_values(y)

        self.__a[0] = np.c_[np.ones(len(X)), X]
        
        if self.opt:
            unraveled_thetas = self.weights[0].ravel()
            for theta in self.weights[1:]:
                unraveled_thetas = np.r_[unraveled_thetas, theta.ravel()]
                
            if self.check_gradients:
                
                print "gradient check with scipy", optimize.check_grad(self.__cost_func_opt, self.__gradf, unraveled_thetas)
                self.__grad_check(unraveled_thetas, X)
            
            theta_opt,min_val,c,d, e = optimize.fmin_cg(self.__cost_func_opt, fprime=self.__gradf, x0 = unraveled_thetas,\
                                                        args = (X, y, m), full_output=1, gtol=1e-5)
            
#             theta_opt= optimize.fmin_bfgs(self.__cost_func_opt, fprime=self.__gradf, x0 = unraveled_thetas,\
#                                                         args = (X,y), gtol=1e-13)
            
#             theta_opt= optimize.fmin_l_bfgs_b( self.__cost_func_opt, fprime=self.__gradf, x0 = unraveled_thetas,\
#                                                         args = (X, y, m), pgtol=1e-10)[0]
            
            self.__convert_weights_back_to_matrix(theta_opt)
            

        else:
            for i in range(self.epochs):
                self.__feed_forward(X)

                current_cost = self.__cost_function_se()
        
                self.cost.append(current_cost)
#                 if abs(self.cost[-1] - self.cost[-2]) < self.__stoping_threshold:
#                     print "cost not decreasing", i
#                     break

                self.__back_prop()

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
        return np.mean(self.predict(X) == y)

if __name__ == '__main__':
    # # xor exmaple
    # X = np.array([[0,0], [0,1], [1,0], [1,1]])
    # y = np.array([[0,1,1,0]]).T
    # nn = Neural_Network([2,2,1], learning_rate=1)
    # nn.fit(X, y)
    # print nn.predict(X)

    X = np.random.rand(300,2)
    y = np.atleast_2d(1 * np.logical_or((X[:,0] - .8) ** 2 + (X[:,1]-.5) ** 2 < .03, (X[:,0] - .2) ** 2 + (X[:,1]-.8) ** 2 < .03)).T

    nn = Neural_Network([2,10,1], learning_rate=1, opt=False, C=0)
    nn.fit(X,y)

    xx, yy = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
    Z = np.round(nn.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,8))
    plt.contourf(xx, yy, Z, alpha=.8)
    plt.scatter(X[:,0], X[:,1], c=y, s=60)
    plt.title("Neural Net Accuracy is " + str(np.round(nn.score(X,y),2)))
    plt.show()
    
