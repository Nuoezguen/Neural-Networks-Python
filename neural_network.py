class Neural_Network():
    
    def __init__(self, layer_sizes, learning_rate=.01):
        self.weights = []
        self.cost = []
        self.__DELTAS = []
        self.__deltas = range(len(layer_sizes) - 1)
        self.__a = range(len(layer_sizes))
        self.epsilon = .15
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
       
        np.random.seed(10)
             
        for i, layer_size in enumerate(layer_sizes[:-1]):
            self.weights.append(self.get_random_weights(layer_sizes[i:i+2]))
            self.__DELTAS.append(np.zeros((layer_size + 1, layer_sizes[i+1])))
                                
    
    def get_random_weights(self, x):
        return np.random.rand(x[0] + 1, x[1]) * 2 * self.epsilon - self.epsilon
        
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def __sigmoid_gradient(self, x):
        return np.multiply(x, 1-x)
    
    def feed_forward(self, X):
        for i, theta in enumerate(self.weights):
            z = np.dot(self.__a[i], theta)
            self.__a[i+1] = np.c_[np.ones(len(z)), self.__sigmoid(z)]
            
        self.__a[-1] = self.__a[-1][:,1:]
    
    def back_prop(self, y):
        self.__deltas[-1] = np.multiply(self.__a[-1] - y, self.__sigmoid_gradient(self.__a[-1]))

        for i in range(-1, -len(self.__deltas), -1):
            self.__deltas[i-1] = np.multiply(np.dot(self.__deltas[i], self.weights[i].T), self.__sigmoid_gradient(self.__a[i-1]))[:, 1:]

        for i, D in enumerate(self.__DELTAS):
            self.__DELTAS[i] += np.dot(self.__a[i].T, self.__deltas[i])
            self.weights[i] -= self.learning_rate * self.__DELTAS[i] / len(y)
      
    
    def predict(self, X):
        a = np.c_[np.ones(len(X)), X]
        for i, theta in enumerate(self.weights):
            z = np.dot(a, theta)
            a = np.c_[np.ones(len(z)), self.__sigmoid(z)]
            
        return a[:,1:]
        
        
    def fit(self, X, y):
        self.__a[0] = np.c_[np.ones(len(X)), X]
    
        for i in range(3000):
            self.feed_forward(X)
            
            current_cost = cost_function(y, self.__a[-1], len(y))
            self.cost.append(current_cost)
    
            self.back_prop(y)

    if __name__ == '__main__':
        X = np.array([[0,0], [0,1], [1,0], [1,1]])
        y = np.array([0,1,1,0]).T
        nn = Neural_Network([2,2,1], learning_rate=.1)
        nn.fit(X, y)
        nn.predict(X)
