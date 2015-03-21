import numpy as np
import matplotlib.pyplot as plt
# generate synetic datasete
def genData(number, ifplot=0, a = 0.5, b = 0.6, r = 0.4):
	x = np.random.uniform(0, 1, number).reshape(number, 1)
	y = np.random.uniform(0, 1, number).reshape(number, 1)
	# print x.shaper
	label = np.zeros((number, 1))
	data = np.concatenate((x, y), axis=1)
	# print data.shape
	# plt.figure()
	for idx, point in enumerate(data):
		# print point.shape
		label[idx] = pow(point[0]-a, 2) + pow(point[1]-b, 2) < pow(r,2)

		# if ifplot:
		# 	if label[idx] == 1:
                # plt.plot(point[0], point[1], 'r+')
	        # else:
				# plt.plot(point[0], point[1], 'b*')

	# if ifplot:
        # plt.show()

	# print label
	return data.T, label.T

if __name__ == "__main__":
	genData(1000)