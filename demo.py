import nn
import synthetic_data
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn import datasets

# X, y = synthetic_data.genData(10000, ifplot=0)

ir = datasets.load_iris()
X, y = ir.data, ir.target

clf = nn.neutral_network(10, maxiter=700)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)

print "begin training"
clf.fit(X_train, y_train)
print "begin testing"
print "Acc: ", clf.evaluate(X_test,y_test)
# accuracy_score(y, predict)

# results = clf.predict(X)
