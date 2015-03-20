import numpy as np
from scipy import optimize

class neutral_network:
	def __init__(self):
		self.layers = 10
		self.output = 2

	def neutral_network(self, hidden_layers, out, regularized = 1):
		self.hidden_layers = hidden_layers
		self.out_dim = out
		slef.regularized = regularized

	def sigmod(self, x):
		return 1 / (1 + np.exp(-z))

	def dsigmod(self, x):
		return x * (1 - x)

	def init_weight(self, in_dim, out_dim):
		# what is the good way initalize
		# do not foget bias term
		return np.random.rand(in_dim+1, out_dim)

	def cat(self, v1, v2):
		return np.concatenate(v1.reshape(-1), v2.reshape(-1))
	def uncat(self, theta, in_dim, hidden_dim):
		dim = in_dim * (1 + hidden_dim)
		t1 = theta(:dim).reshape((in_dim+1, hidden_dim))
		t2 = theta(dim:).shape((hidden_dim+1,1))
		return t1, t2

	def fit(self, X, y):
		# X is in n x p format
		# y shou
		self.nfeatures = X.shape[0]
		self.nsamples = X.shape[1]

		theta1 = self.init_weight(nfeatures, self.hidden_layers)
		theta2 = self.init_weight(self.hidden_layers, out)
		theta = self.cat(theta1, theta2)

		_res = optimize.minimize(self.function, thetas0, jac=self.function_prime, method=self.method, 
                                 args=(nfeatures, self.hidden_layers, self.nsamples, X, y), options=options)
        
		self.t1, self.t2  = self.uncat(_res.x, in_dim, hidden_dim)



	def _forward(self, X, t1, t2):
		# X is in n * p
		nfeatures = X.shape[0]
		samples = X.shape[1]
		hidden_layers = t1.shape[0]

		term1 = np.ones(nfeatures).reshape(samples, 1)
		term2 = np.ones(hidden_layers).reshpae(hidden_layers, 1)
		X_plus = np.concatenate((term1, X), axis=0) # make it to n+1 x p

		# hidden layer
		z2 = np.dot(t1.T, X_plus)
		a2 = sigmod(z2)

		#output
		a2_plus = np.concatenate((term2, a2), axis=0)
		z3 = np.dot(t2.T, a2_plus)
		a3 = sigmond(z3)

		return x_plus, a2.T, z2.T, z3.T, a3.T

	def CostFunction(self, theta, in_dim, hidden_dim, num_labels, X, y):
		# compute cost function
		m = X.shape[1] #number of samples
		t1, t2 = self.uncat(theta, theta, in_dim, hidden_dim)

		o = self._forward(X, t1, t2)
		J = np.power((o - y), 2)
		J = np.sum(J) / m

		return J

	def dCostFunction(self, theta, in_dim, hidden_dim, num_labels, X, y):
		#compute gradient
		t1, t2 = self.uncat(theta, theta, in_dim, hidden_dim)

		t1 = t1[:, 1:] # remove bias term
		t2 = t2[:, 1:] 
		a1, z2, a2, z3, a3 = _self._forward(X, t1, t2) # p x s matrix
		sigma3 = -(y - a3)

		sigma2 = np.dot(t2, sigma3) * dsigmod(a3)

		theta2_grad = np.dot(sigma3.T, a2)
		theta1_grad = np.dot(sigma2.T, a1)
		
		theta1_grad = theta1 / m
		theta2_grad = theta2 / m

		return self.cat(theta1, theta2)

	def predict(self, X):
		_,_,_,_, out = self._forward(X, self.t1, self.t2)
		return out, out.argmax(0)

	