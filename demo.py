import nn
import circle_data
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn import datasets
import numpy as np
def visualize(clf, plt):
	W = clf.t1.T
	x = np.linspace(0,1,100)
	for w in W:
		y = -1 * (w[1] * x + w[0]) / w[2]
		plt.plot(x,y)
	plt.axis((0 ,1, 0, 1))
	plt.show()

d = circle_data.circle_data(1000, ifplot=1)

# ir = datasets.load_iris()
X, y, plt = d.data, d.target, d.plt

clf = nn.neutral_network(10, epoch=10, maxiter=300, activate="sigmoid", show=0)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)

clf.fit(X_train, y_train)

visualize(clf, plt)
print "Acc: ", clf.evaluate(X_test,y_test)

