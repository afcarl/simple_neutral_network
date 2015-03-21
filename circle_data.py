import numpy as np
import matplotlib.pyplot as plt
# generate synetic datasete

class circle_data:
	def __init__(self, num=1000,ifplot=0):
		self.data, self.target = self.genData(num, ifplot)

	def genData(self, number, ifplot, a = 0.5, b = 0.6, r = 0.4):
		data = np.random.uniform(0, 1, 2*number).reshape((number, 2))

		# print x.shaper
		label = np.zeros((number), dtype='int64')
		# print data.shape
		# plt.figure()
		for idx, point in enumerate(data):
			# print point.shape
			label[idx] = pow(point[0]-a, 2) + pow(point[1]-b, 2) < pow(r,2)

			if ifplot:
				if label[idx] == 1:
					plt.plot(point[0], point[1], 'r+')
				else:
					plt.plot(point[0], point[1], 'b*')

		if ifplot:
			plt.show()

		# print label
		return data, label

if __name__ == "__main__":
	genData(1000)