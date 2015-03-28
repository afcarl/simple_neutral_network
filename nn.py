import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
class neutral_network:

	def __init__(self, hidden_layers,  epoch=1, regularized=1, maxiter=500, show=1, activate='sigmoid'):
		self.hidden_layers = hidden_layers
		# self.out_dim = out
		self.regularized = regularized
		# self.maxiter = 100
		self.eplison = 0.12
		self.maxiter = maxiter
		self.method = 'TNC'
		self.iter = 0
		self.epochs = epoch
		self.show = show
		self.activationfun = activate
		

	def activation(self, x):
		if self.activationfun == "sigmoid":
			return 1 / (1 + np.exp(-x))
		if self.activationfun == "step":
			return  np.sign(x) + 1

	def dactivation(self, x):

		if self.activationfun == "sigmoid":
			x = self.activation(x)
			x = x * (1 - x)
		if self.activationfun == "step":
			x = np.logical_xor(x, np.ones(x.shape))
			x = 1 * x
			print x
		return x

	def init_weight(self, in_dim, out_dim):
		# what is the good way initalize
		# do not foget bias term
		return np.random.rand(in_dim+1, out_dim) * 2 * self.eplison - self.eplison

	def cat(self, v1, v2):
		return np.concatenate((v1.reshape(-1), v2.reshape(-1)))

	def uncat(self, theta, in_dim, hidden_dim):
		dim = (in_dim+1) * hidden_dim
		t1 = theta[:dim].reshape((in_dim+1, hidden_dim))
		t2 = theta[dim:].reshape((hidden_dim+1, self.classes))
		return t1, t2
	def CostFunction(self, theta, in_dim, hidden_dim, num_labels, X, y):
		# compute cost function
		t1, t2 = self.uncat(theta, in_dim, hidden_dim)

		_, _, _, _, o = self._forward(X, t1, t2)
		J = np.power((o - y), 2)
		J = np.sum(J) / self.nsamples
		
		# print self.iter, J
		# self.iter += 1
		return J

	def dCostFunction(self, theta, in_dim, hidden_dim, num_labels, X, y):
		#compute gradient
		t1, t2 = self.uncat(theta, in_dim, hidden_dim)


		a1, z2, a2, z3, a3 = self._forward(X, t1, t2) # p x s matrix

		# t1 = t1[1:, :] # remove bias term
		# t2 = t2[1:, :]
		sigma3 = -(y - a3) * self.dactivation(z3) # do not apply dsigmode here? should I
		sigma2 = np.dot(t2, sigma3)
		term = np.ones((1,num_labels))
		sigma2 = sigma2 * np.concatenate((term, self.dactivation(z2)),axis=0)

		theta2_grad = np.dot(sigma3, a2.T)
		theta1_grad = np.dot(sigma2[1:,:], a1.T)

		theta1_grad = theta1_grad / num_labels
		theta2_grad = theta2_grad / num_labels

		return self.cat(theta1_grad.T, theta2_grad.T)

	def labelMarix(self, y):
		y = y.reshape((self.nsamples,))
		Y = np.zeros((self.classes, len(y)), dtype='int64')

		for (idx, p) in enumerate(y):
			Y[p][idx] = 1
		return Y

	def fit(self, X, y, X_t, y_t):
		# X is in n x p format
		# y shou
		self.nfeatures = X.shape[1]
		self.nsamples = X.shape[0]
		self.classes = len(np.unique(y))
		Y = self.labelMarix(y)

		theta1 = self.init_weight(self.nfeatures, self.hidden_layers)
		theta2 = self.init_weight(self.hidden_layers, self.classes)
		theta = self.cat(theta1, theta2)
		self.error_v = np.zeros((self.epochs,))
		self.acc = np.zeros((2,self.epochs))
		# every epoch
		print "begin training"
		print "--max iteration: ", self.maxiter
		for i in range(self.epochs):
			

			J = self.CostFunction(theta, self.nfeatures, self.hidden_layers, self.nsamples, X, Y)
			options = {'maxiter': self.maxiter}
			_res = optimize.minimize(self.CostFunction, theta, jac=self.dCostFunction, method=self.method,
								 args=(self.nfeatures, self.hidden_layers, self.nsamples, X, Y), options=options)
			print "epoch %d/%d" % (i+1, self.epochs), ": errors: ", _res.fun  
			self.error_v[i] = _res.fun

			self.t1, self.t2  = self.uncat(_res.x, self.nfeatures, self.hidden_layers)
			self.acc[0][i] = self.evaluate(X, y)
			self.acc[1][i] = self.evaluate(X_t, y_t)
			theta = _res.x
		
		# if self.show: 
			# self.visualize()

		
		#print self.t1
		#print self.t2
		
	def visualize(self, iter):
		plt.figure(2)
		plt.subplot(211)
		plt.title('simple_neutral_network')   # subplot 211 title
		plt.xlabel('# of epoch')
		plt.ylabel('error cost')
		plt.plot(self.error_v,'r', linewidth=5)
		
		
		plt.subplot(212)
		plt.plot(self.acc[0], linewidth=3, label='train_acc')
		plt.plot(self.acc[1], linewidth=3, label='test_acc')
		plt.legend(loc=3, ncol=2, borderaxespad=0.)

		plt.savefig(iter + 'epochs.png')
		plt.close(2)
		# plt.show()

	def _forward(self, X, t1, t2):
		# X is in p * n
		X = X.T # conver to n * p
		term1 = np.ones(self.nsamples).reshape(1, self.nsamples)
		a1 = np.concatenate((term1, X), axis=0) # make it to n+1 x p

		# hidden layer
		z2 = np.dot(t1.T, a1)
		a2 = self.activation(z2)

		#output
		a2 = np.concatenate((term1, a2), axis=0) # first add activation and then add bias term
		z3 = np.dot(t2.T, a2)
		a3 = self.activation(z3)

		return a1, z2, a2, z3, a3

	def predict(self, X):
		_, _, _, _, out = self._forward(X, self.t1, self.t2)
		return out, out.argmax(0)

	def evaluate(self, X, y):
		# print "begin evaluation"
		prob, label = self.predict(X)
		acc = 0.0;
		y = y.reshape((len(label),))
		for i in range(len(y)):
			if y[i] == label[i]:
				acc += 1
		return  acc / len(y)
		# return prob, label






