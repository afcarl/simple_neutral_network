import nn
import circle_data
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn import datasets

d = circle_data.circle_data(10000)

# ir = datasets.load_iris()
X, y = d.data, d.target

clf = nn.neutral_network(10, epoch=10, maxiter=300, activate="sigmoid")
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)


clf.fit(X_train, y_train)

print "Acc: ", clf.evaluate(X_test,y_test)
# accuracy_score(y, predict)

# results = clf.predict(X)
