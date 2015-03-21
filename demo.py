import nn
import synthetic_data

X, y = synthetic_data.genData(100, ifplot=0)

clf = nn.neutral_network(10, 1)
clf.fit(X, y)

results = clf.predict(X)
