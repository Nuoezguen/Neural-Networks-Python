import numpy as np


class Feed_Forward:

	def __init__(self, activation_func="sigmoid"):
		if activation_func == "sigmoid":
			self.activation_func = self.__sigmoid
		if activation_func == "tanh":
			self.activation_func = self.__tanh
		if activation_func == "step":
			self.activation_func = self.__step_func

	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def __tanh(self, x):
		return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

	def __step_func(self, x, threshold=0):
		return (x > threshold) * 1

	def __add_bias(self, X):
		return np.concatenate((np.ones((len(X), 1)), X), axis=1)

	def __feed(self, X, theta):
		return self.activation_func(np.dot(self.__add_bias(X), theta))

	def feed_forward(self, X, thetas):
		'''
		Loop through all layers with current weight matrix(theta)

		Returns final predicition
		'''
		y_pred = X.copy()
		for theta in thetas:
			y_pred = self.__feed(y_pred, theta)
		return y_pred

##### END Class #####


def xor(row):
	x1, x2 = row
	if x1 > .5 and x2 < .5:
		return 1
	if x1 < .5 and x2 > .5:
		return 1
	return 0

if __name__ == '__main__':
	'''
	Some dummy data in R2 [0, 1] x [0, 1] is created where points with (x1<.5 and x2>.5)
	or (x1>.5 and x2<.5) are labeled 1. This is a classic xor example.
	There are a total of 4 layers
	
	Input layer - random x's
	First hidden layer -  (2 units) transform all x's to one of four points with x1 and x2 either 0 or 1
	Second hidden layer - (2 units) map point 1,1 to 0,0 with first unit being And(not x1, x2) and the other
							being And(x1, not x2)
	Output - now data is linearly separable

	Thetas - There are three different weight matrices for each layer jump. All matrices have 3 rows,
			one for the bias unit and 2 for x1, x2. The first two weight matrices have two units and the last has 
			one because there is only 2 classes (0 or 1)

	'''
	X = np.random.rand(100, 2)
	y = np.apply_along_axis(xor, 1, X)
	y.shape = (len(y), 1)
	thetas = [np.array([[-10, -10], [20, 0], [0, 20]]), np.array([[-20, -20], [-30, 30], [30, -30]]),
          np.array([[-2], [3], [3]])]
	ff = Feed_Forward("step")
	y_pred = ff.feed_forward(X, thetas)
	print "Accuracy of xor", np.mean(y == y_pred)


