import numpy as np
import csv 
import matplotlib.pyplot as plt
import time
from collections import Counter
import sys

# Testing
from sklearn.metrics import accuracy_score

#==============================================================================
# Useful Constants
#==============================================================================
num_iterations = 1000

neighborhood = 8 / 2.0
time_constant = num_iterations / np.log(neighborhood)
# Every dt iterations, update plotting
dt = 100

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
		self.learning_rate = 0.1
		self.som_size = width, height
		self.cell_size = len(dataset[0])
		self.cells = np.random.random((width, height, self.cell_size))

	def clear(self):
		self.cells = np.random.random((self.som_size[0], self.som_size[1], self.cell_size))

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
		return self.learning_rate * np.exp(- i / time)

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
		        w = self.cells[x, y, :-1].reshape(self.cell_size-1, 1)
		        # Don't bother with actual Euclidean distance, to avoid expensive sqrt operation
		        sq_dist = np.sum((w - t) ** 2)
		        if sq_dist < min_dist:
		            min_dist = sq_dist
		            bmu_idx = np.array([x, y])
		# Get prototype corresponding to bmu_idx
		bmu = self.cells[bmu_idx[0], bmu_idx[1], :-1].reshape(self.cell_size-1, 1)
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
				w = self.cells[x, y, :-1].reshape(self.cell_size-1, 1)
				# Get the 2-D distance (again the Euclidean distance without sqrt)
				w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
				# If the distance is within the current neighbourhood radius
				if w_dist <= radius**2:
					# Calculate the degree of influence
					influence = self.calculate_gaussian_influence(w_dist, radius)
					# Now update the prototypes
					new_w = w + (alpha * influence * (example - w))
					# Commit the new prototype, don't forget to "switch" dimensions
					self.cells[x, y, :-1] = new_w.reshape(1, self.cell_size-1)		

	def train(self, n):
		"""
		Train SOM over n-th iterations

		Parameters:
			n: iterations' number
		"""
		for i in range(n):
			self.train_one_time(i)


	def draw(self, ax, **kwargs):
		# Displaying Dataset
		x1, y1 = [], []
		x2, y2 = [], []
		x3, y3 = [], []
		p = 2 # petal length
		q = 3 # petal width 
		ax.clear()
		for i in range(8):
			for j in range(8):
				if net.cells[i, j, -1] == 1:
					x1.append(net.cells[i, j, p])
					y1.append(net.cells[i, j, q])
				elif net.cells[i, j, -1] == 2:
					x2.append(net.cells[i, j, p])
					y2.append(net.cells[i, j, q])
				elif net.cells[i, j, -1] == 3:
					x3.append(net.cells[i, j, p])
					y3.append(net.cells[i, j, q])
		type1 = ax.scatter(x1, y1, s=50, c='red')
		type2 = ax.scatter(x2, y2, s=50, c='green')
		type3 = ax.scatter(x3, y3, s=50, c='blue')

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
		type11 = ax.scatter(x1, y1, s=25, c='red', marker="+")
		type21 = ax.scatter(x2, y2, s=25, c='green', marker="+")
		type31 = ax.scatter(x3, y3, s=25, c='blue', marker="+")

		ax.set_title('Base Iris', fontsize=14)
		ax.set_xlabel('Longueur petal (cm)')
		ax.set_ylabel('Largeur petal (cm)')
		ax.legend([type1, type2, type3, type11, type21, type31 ], 
			["Prototype Setosa", "Prototype Versicolor", "Prototype Virginica",
			"Iris Setosa", "Iris Versicolor", "Iris Virginica" ], 
			#loc=1, 
			bbox_to_anchor=(1.0, 1.0))
		return ax

	def assignNeighbor(self, k):
		n, m = self.som_size
		self.cells = self.cells.reshape([n*m, 5])
		predictions = []
		kNearestNeighbor(self.data[:, 0:4], self.data[:, 4], self.cells[:, :-1], predictions, k)
		for i in range(len(self.cells)):
			self.cells[i, 4] = predictions[i]
		self.cells = self.cells.reshape([n, m, 5])		

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
	# Loop over all observations
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test[i, :], k))

def confusion_matrix(y_real, y_predicted):
	matrix = np.zeros([len(set(y_real)), len(set(y_real))])
	for i in range(len(set(y_real))):
		for j in range(len(set(y_real))):
			matrix[i, j] = len(
				filter(
					lambda x : x[0] == i+1 and x[1] == j+1, 
					zip(y_real, y_predicted)
					)
				)
	return matrix

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

	net.assignNeighbor(5)

	fig = plt.figure()
	plt.ion()
	ax = fig.add_subplot(111)

	accuracy 	= 0
	bloc_size 	= 30
	np.random.shuffle(X)

	# Cross-Validation
	for i in range(5):
		accuracy 	= 0
		x_test 		= X[bloc_size * i : bloc_size * (i+1)]
		indexes 	= range(i * bloc_size, (i+1) * bloc_size) 

		x_remain 	= np.delete(X, indexes, 0)

		for j in range(4):
			# Split datas
			x_valid = x_remain[j * bloc_size:(j+1) * bloc_size]
			indexes = range(j * bloc_size, (j+1) * bloc_size) 
			x_train = np.delete(x_remain, indexes, 0)

			# Clear previous learning
			net.data = x_train
			net.clear()

			# Start Learning
			for t in range(num_iterations):
				net.train_one_time(t)
				cond = (t == 0) or (t == 75) or (t == 300) or (t == num_iterations-1)
				if cond:
					net.assignNeighbor(5)
					ax.clear()
					ax = net.draw(ax)
					plt.plot()
					plt.draw()
					plt.pause(10.0)
					#fig.canvas.draw()
					fig.savefig("kohonen_{}".format(t))

			# KNN assignement
			net.assignNeighbor(5)

			# Accuracy calculus
			y_valid = x_valid[:, -1]
			y_predicted = []
			for x in x_valid:
				_, bmu_idx = net.find_bmu(x)
				y_predicted.append(net.cells[bmu_idx[0], bmu_idx[1], -1])

			# Testing
			#print y_valid, y_predicted
			accuracy += accuracy_score(y_valid, y_predicted)

			#matrix = confusion_matrix(y_valid, y_predicted)
			#print matrix
			#accuracy += sum([matrix[n, n] for n in range(len(matrix))]) / matrix.sum()
			#print "Number of 1 : {}".format(sum(sum(net.cells[:, :, -1] == 1)))
			#print "Number of 2 : {}".format(sum(sum(net.cells[:, :, -1] == 2)))
			#print "Number of 3 : {}".format(sum(sum(net.cells[:, :, -1] == 3)))
		print "Accuracy iteration {} : {}".format(i, accuracy / 4)
		