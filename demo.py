import nn
import synthetic_data

X, y = synthetic_data.genData(100)

clf = nn.neutral_network(10, 1)

