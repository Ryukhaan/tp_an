import numpy as np
import csv 
import matplotlib.pyplot as plt
import time

#==============================================================================
# Useful Constants
#==============================================================================
num_iterations = 1000
neighborhood = max(som_size[0], som_size[1]) / 2.0
time_constant = num_iterations / np.log(neighborhood)
# Every dt iterations, update plotting
dt = 10

#==============================================================================
# ** Self Organized Map
#------------------------------------------------------------------------------
#  This class performs SOM.
#==============================================================================
class SOM(object):
	"""This class performs SOM
	Attributes:
		data 			: Will be IRIS dataset
		radius 			: Gaussian std
		learning_rate 	: Learning rate
		som_size 		: Will be 8x8
		cell_size 		: Size of a prototype (here 4)
		cells 			: Prototypes
	"""
	def __init__(self, width, height, dataset):
		super(SOM, self).__init__()

		self.data = dataset
		self.radius = max(width, height) / 2.0
		self.learning_rate = 0.12
		self.som_size = width, height
		self.cell_size = len(dataset[0]) - 1
		self.cells = np.random.random((width, height, self.cell_size))

	def decay_radius(self, i, time = time_constant):
		"""
		Parameters:
			i : int

		Return:
			Gaussian std at the i-th iterations
		"""
		return self.radius * np.exp(-i / time)

	def decay_learning_rate(self, i, time = time_constant):
		"""
		Parameters:
			i 	: int
			time: int

		Return:
			learning rate at the i-th iterations
		"""
		return self.learning_rate * np.exp(-i / time)

	def calculate_gaussian_influence(self, distance, radius):
		"""
		Parameters:
			distance 	: float
			radius 		: float

		Return:
			how the gaussian momdifies prototypes' update according theirs distance to the bmu
		"""
		return np.exp(-distance / (2* (radius**2)))

	def find_bmu(self, t):
		"""
		Parameters:
			t : an example from self.data

		Returns:
			bmu 	: bmu's prototypes
			bmu_idx : index of the bmu
		"""
		bmu_idx = np.array([0, 0])
		# Set the initial minimum distance to a huge number
		min_dist = np.iinfo(np.int).max
		# Calculate the distance between each prototype and the input
		for x in range(self.som_size[0]):
		    for y in range(self.som_size[1]):
		        w = self.cells[x, y, :].reshape(self.cell_size, 1)
		        # Don't bother with actual Euclidean distance, to avoid expensive sqrt operation
		        sq_dist = np.sum((w - t) ** 2)
		        if sq_dist < min_dist:
		            min_dist = sq_dist
		            bmu_idx = np.array([x, y])
		# Get prototype corresponding to bmu_idx
		bmu = self.cells[bmu_idx[0], bmu_idx[1], :].reshape(self.cell_size, 1)
		# Return the (bmu, bmu_idx) tuple
		return (bmu, bmu_idx)

	def train_one_time(self, i):
		"""
		Training one iteration

		Parameters:
			i:	iteration's number 
		"""
		# Get a training example
		example = self.data[np.random.randint(0, len(self.data)), 0:4].reshape(np.array([4, 1]))

		# Find its BMU
		bmu, bmu_idx = self.find_bmu(example)

		# Get new SOM's paramaters
		radius = self.decay_radius(i)
		alpha = self.decay_learning_rate(i, num_iterations)

		# Update prototypes
		for x in range(self.som_size[0]):
			for y in range(self.som_size[1]):
				w = self.cells[x, y, :].reshape(self.cell_size, 1)
				# Get the 2-D distance (again the Euclidean distance without sqrt)
				w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
				# If the distance is within the current neighbourhood radius
				if w_dist <= radius**2:
					# Calculate the degree of influence
					influence = self.calculate_gaussian_influence(w_dist, radius)
					# Now update the prototypes
					new_w = w + (alpha * influence * (example - w))
					# Commit the new prototype, don't forget to "switch" dimensions
					self.cells[x, y, :] = new_w.reshape(1, self.cell_size)		

	def train(self, n):
		"""
		Train SOM over n-th iterations

		Parameters:
			n: iterations' number
		"""
		for i in range(n):
			self.train_one_time(i)

def predict(X_train, y_train, x_test, k):

	# create list for distances and targets
	distances = []
	targets = []

	# Can it be done with intention list ?
	for i in range(len(X_train)):
		# Euclidean distance
		distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
		# Add it to list of distances
		distances.append([distance, i])

	# Sort the list of distances
	distances = sorted(distances)

	# List of the k neighbors
	for i in range(k):
		index = distances[i][1]
		targets.append(y_train[index])

	# Return most common target
	return Counter(targets).most_common(1)[0][0]


def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
	# train, is it really mandatory ?
	train(X_train, y_train)

	# loop over all observations
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test[i, :], k))

if __name__ == "__main__":
	# Creating Dataset
	X = list()
	with open('./iris_norm.txt', 'rb') as file :
		spamreader = csv.reader(file, delimiter=',', quotechar='|')
		for row in spamreader:
			if not row: continue
			tmp = []
			for elem in row[0:5]:
				tmp.append(float(elem))
			X.append(tmp)
			#y.append(row[-1])
	X = np.asarray(X)
	
	# Creating Self-Organasing Map
	net = SOM(width = 8, height = 8, dataset = X)

	# Displaying Dataset
	x1, y1 = [], []
	x2, y2 = [], []
	x3, y3 = [], []
	p = 2 # petal length
	q = 3 # petal width 
	for i, elem in enumerate(net.data):
		if net.data[i, -1] == 1:
			x1.append(net.data[i][p])
			y1.append(net.data[i][q])
		elif net.data[i, -1] == 2:
			x2.append(net.data[i][p])
			y2.append(net.data[i][q])
		elif net.data[i, -1] == 3:
			x3.append(net.data[i][p])
			y3.append(net.data[i][q])

	x_net = net.cells[:, :, p].reshape(np.array([net.som_size[0] * net.som_size[1], 1]))
	y_net = net.cells[:, :, q].reshape(np.array([net.som_size[0] * net.som_size[1], 1]))

	fig = plt.figure()
	plt.ion()
	#plt.hold(False)
	ax = fig.add_subplot(111)
	type1 = ax.scatter(x1, y1, s=50, c='red')
	type2 = ax.scatter(x2, y2, s=50, c='green')
	type3 = ax.scatter(x3, y3, s=50, c='blue')
	type4 = ax.scatter(x_net, y_net, s=50, c='black')
	ax.set_title('Taille du petal : jeu de donnees Iris', fontsize=14)
	ax.set_xlabel('Longueur petal  normalisee (cm)')
	ax.set_ylabel('Largeur petal normalisee (cm)')
	ax.legend([type1, type2, type3, type4], ["Iris Setosa", "Iris Versicolor", "Iris Virginica", "Prototypes"], loc=2)
	ax.grid(True,linestyle='-',color='0.75')
	plt.plot()
	plt.pause(5)

	# Start
	net.train_one_time(0)
	x_net = net.cells[:, :, p].reshape(np.array([net.som_size[0] * net.som_size[1], 1]))
	y_net = net.cells[:, :, q].reshape(np.array([net.som_size[0] * net.som_size[1], 1]))
	type4 = ax.scatter(x_net, y_net, s=50, c='black')
	ax.legend([type1, type2, type3, type4], ["Iris Setosa", "Iris Versicolor", "Iris Virginica", "Cells"], loc=2)
	ax.grid(True,linestyle='-',color='0.75')
	plt.plot()
	plt.draw()

	# Learning
	for i in range(num_iterations):
		#time.sleep(0.1)
		net.train_one_time(i+1)
		if i % dt == 0:
			ax.clear()
			x_net = net.cells[:, :, p].reshape(np.array([net.som_size[0] * net.som_size[1], 1]))
			y_net = net.cells[:, :, q].reshape(np.array([net.som_size[0] * net.som_size[1], 1]))
			type1 = ax.scatter(x1, y1, s=50, c='red')
			type2 = ax.scatter(x2, y2, s=50, c='green')
			type3 = ax.scatter(x3, y3, s=50, c='blue')
			type4 = ax.scatter(x_net, y_net, s=50, c="black")
			#ax.legend([type4])
			plt.plot()
			plt.draw()
			plt.pause(0.01)
			#fig.canvas.draw()

	x_net = net.cells[:, :, p].reshape(np.array([net.som_size[0] * net.som_size[1], 1]))
	y_net = net.cells[:, :, q].reshape(np.array([net.som_size[0] * net.som_size[1], 1]))
	type4 = ax.scatter(x_net, y_net, s=50, c='yellow')
	#ax.legend([type1, type2, type3, type4, type5], ["Iris Setosa", "Iris Versicolor", "Iris Virginica", "Cells"], loc=2)
	#ax.grid(True,linestyle='-',color='0.75')
	plt.plot()


		