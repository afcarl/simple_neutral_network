import nn
import circle_data
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
global iter
iter = 0
global acc
acc = [0 for i in range(10)]
def visualize(clf):
	W = clf.t1.T
	x = np.linspace(0,1,100)
	plt.figure(1) #save as circle_data
	for w in W:
		y = -1 * (w[1] * x + w[0]) / w[2]
		plt.plot(x,y)
	plt.axis((0 ,1, 0, 1))
	
	plt.savefig(str(iter) + '.png')
	plt.close(1)

def main():
	d = circle_data.circle_data(1000, ifplot=1)

	# ir = datasets.load_iris()
	X, y = d.data, d.target

	clf = nn.neutral_network(10, epoch=10, maxiter=300, activate="sigmoid", show=1)

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)

	clf.fit(X_train, y_train, X_test, y_test)
	clf.visualize(str(iter))

	visualize(clf)
	acc[iter] = clf.evaluate(X_test,y_test)
	


if __name__ == "__main__":
	for i in range(10):
		main()
		iter = iter + 1
	print "Acc: ", acc

	plt.figure(3)
	plt.xlabel('# of iteration')
	plt.ylabel('accuracy')
	plt.plot(acc,linewidth=5)
	plt.savefig('acc.png')