import numpy as np
from scipy import optimize

class neutral_network:

	def __init__(self, hidden_layers, out, regularized = 1):
		self.hidden_layers = hidden_layers
		self.out_dim = out
		self.regularized = regularized

	def sigmod(self, x):
		return 1 / (1 + np.exp(-x))

	def dsigmod(self, x):
		return x * (1 - x)

	def init_weight(self, in_dim, out_dim):
		# what is the good way initalize
		# do not foget bias term
		return np.random.rand(in_dim+1, out_dim)

	def cat(self, v1, v2):
		return np.concatenate((v1.reshape(-1), v2.reshape(-1)))

	def uncat(self, theta, in_dim, hidden_dim):
		dim = (in_dim+1) * hidden_dim
		t1 = theta[:dim].reshape((in_dim+1, hidden_dim))
		t2 = theta[dim:].reshape((hidden_dim+1,1))
		return t1, t2

	def fit(self, X, y):
		# X is in n x p format
		# y shou
		self.nfeatures = X.shape[0]
		self.nsamples = X.shape[1]

		theta1 = self.init_weight(self.nfeatures, self.hidden_layers)
		theta2 = self.init_weight(self.hidden_layers, y.shape[0])
		theta = self.cat(theta1, theta2)
		J = self.CostFunction(theta, self.nfeatures, self.hidden_layers, self.nsamples, X, y)
		print J
        grad = self.dCostFunction(theta, self.nfeatures, self.hidden_layers, self.nsamples, X, y)

		_res = optimize.minimize(self.function, theta, jac=self.function_prime, method=self.method,
                                 args=(self.nfeatures, self.hidden_layers, self.nsamples, X, y), options=options)
        
		self.t1, self.t2  = self.uncat(_res.x, self.nfeatures, self.hidden_layers)

	def _forward(self, X, t1, t2):
		# X is in n * p
		nfeatures = X.shape[0]
		nsamples = X.shape[1]
		hidden_layers = t1.shape[1]

		term1 = np.ones(nsamples).reshape(1, nsamples)
		term2 = np.ones(hidden_layers).reshape(hidden_layers, 1)
		X_plus = np.concatenate((term1, X), axis=0) # make it to n+1 x p

		# hidden layer
		z2 = np.dot(t1.T, X_plus)
		a2 = self.sigmod(z2)

		#output
		a2_plus = np.concatenate((term1, a2), axis=0)
		z3 = np.dot(t2.T, a2_plus)
		a3 = self.sigmod(z3)

		return X_plus, a2.T, z2.T, z3.T, a3.T

	def CostFunction(self, theta, in_dim, hidden_dim, num_labels, X, y):
		# compute cost function
		m = X.shape[1] #number of samples
		t1, t2 = self.uncat(theta, in_dim, hidden_dim)

		_, _, _, _, o = self._forward(X, t1, t2)
		J = np.power((o - y.T), 2)
		J = np.sum(J) / m

		return J

	def dCostFunction(self, theta, in_dim, hidden_dim, num_labels, X, y):
		#compute gradient
		t1, t2 = self.uncat(theta, theta, in_dim, hidden_dim)

		t1 = t1[:, 1:] # remove bias term
		t2 = t2[:, 1:] 
		a1, z2, a2, z3, a3 = self._forward(X, t1, t2) # p x s matrix
		
		sigma3 = -(y - a3)
		sigma2 = np.dot(t2, sigma3) * self.dsigmod(a3)

		theta2_grad = np.dot(sigma3.T, a2)
		theta1_grad = np.dot(sigma2.T, a1)
		
		theta1_grad = theta2_grad / m
		theta2_grad = theta2_grad / m

		return self.cat(theta1_grad, theta2_grad)

	def predict(self, X):
		_, _, _, _, out = self._forward(X, self.t1, self.t2)
		return out
	