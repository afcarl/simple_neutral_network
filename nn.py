import numpy as np
from scipy import optimize

class neutral_network:
	def __init__(self):
		self.layers = 10
		self.output = 2

	def neutral_network(self, hidden_layers):
		self.hidden_layers = hidden_layers
	def sigmod(self, x):
		return 1 / (1 + np.exp(-z))
	def dsigmod(self, x):
		return sigmod(x) * (1 - sigmod(x))

	def init_weight(self):
		# what is the good way t
		return np.random.rand(self.nfeatures, self.hidden_layers + 1)

	def fit(self, X, y):
		# X is in n x p format
		# y shou
		self.nfeatures = X.shape[0]
		self.nsamples = X.shape[1]

		theta1_o = self.

	def computeCostFunction(self, X, y):

