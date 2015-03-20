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
		return sigmod(x) * (1 - sigmod(x))

	def init_weight(self, in_dim, out_dim):
		# what is the good way initalize
		# do not foget bias term
		return np.random.rand(in_dim+1, out_dim)

	def cat(self, v1, v2):
		return np.concatenate(v1.reshape(-1), v2.reshape(-2))
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

	def CostFunction(self, theta, in_dim, hidden_dim, num_labels, X, y):
		# compute cost function
		t1, t2 = self.uncat(theta)

		
	def dCostFunction(self, theta, in_dim, hidden_dim, num_labels, X, y):
		#compute gradient
