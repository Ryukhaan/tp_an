import random as random
import math as math
import numpy as np 
import csv as csv
import json

# For style
import pprint

NCN = float('Inf')

# Load network.json
def load_network():
	with open('./network.json', 'rb'):
		network = json.load(open('./network.json', 'rb'))['network']
		for k, layer in enumerate(network):
			for i, neuron in enumerate(layer):
				for key in neuron['weights'].keys():
					if key == 'bias': continue
					for j in range(len(neuron['weights'][key])):
						if neuron['weights'][key][j] == "NCN":
							neuron['weights'][key][j] = NCN
				if neuron['function'] == "None":
					neuron['function'] = None
	return network

# Load file
def load_file():
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
	return np.asarray(X)

# Cross Validation
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = random.randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	equal = lambda x : x[0] == x[1]
	correct = len(filter(equal, zip(actual, predicted)))
	return correct / float(len(actual)) * 100.0

# Calculate neuron activation for a given input
def activate(weights, inputs):
	activation = weights['bias'] # bias ?
	#activation += sum([weights[i] * inputs[i] for i in range(len(weights)-1)]) # normal activation
	for key, value in weights.iteritems():
		# Key is kappa
		# Value is an array with connection with layer kappa 
		if key == 'bias': continue
		for i in range(len(value)):
			if value[i] == NCN: continue
			activation += value[i] * inputs[int(key)][i]
	return activation

# Calculate transfer function
def transfer(activation, function = None):
	if function == None:
		return activation
	elif function == "Tanh":
		return math.tanh(activation)
	elif function == "ReLU":
		return max(0.0, activation)
	elif function == "Sigm":
		return 1.0 / (1.0 + math.exp(-activation))

# Calculate drivative function
def derivated_transfer(output, function = None):
	if function == None:
		return 1.0
	elif function == "Tanh":
		return 1.0 - output * output
	elif function == "ReLU":
		if output < 0.0: return 0.0
		return 1.0
	elif function == "Sigm":
		return output * (1.0 - output)	

# Make forward and calculate activation
def forward_propagate(network, row):
	# Row is the initial input value
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation, function = neuron['function'])
			new_inputs.append(neuron['output'])
		inputs.append(new_inputs)
		#print inputs
	return inputs

# Make backward and calulate error (epsilon)
def backward_propagate(network, expected):
	# For each layer
	for k in reversed(range(len(network))):
		layer = network[k]
		errors = list()
		# Last layer
		if k == len(network) - 1 :
			# For each neuron in the last layer
			for i in range(len(layer)):
				neuron = layer[i]
				errors.append(expected[i] - neuron['output'])
				neuron['epsilon'] = errors[i] * derivated_transfer(neuron['output'], neuron['function'])
		# All others layers
		else:
			# For each neuron
			for i in range(len(layer)):
				#print "Neuron number : ", i
				neuron = layer[i]
				tmp_error = 0.0
				for kappa in range(k+1, len(network)):
					for next_neuron in network[kappa]:
						for key in next_neuron['weights'].keys():
							if key == 'bias': continue
							if int(key) != k+1: continue
							if next_neuron['weights'][key][i] == NCN: continue
							#print "Start", next_neuron['epsilon']
							tmp_error += next_neuron['weights'][key][i] * next_neuron['epsilon']
				errors.append(tmp_error)
				neuron['epsilon'] = errors[i] * derivated_transfer(neuron['output'], neuron['function'])

# Update weights
def update_weights(network, inputs, alpha, momentum, shared):

	# For each layer
	for k, layer in enumerate(network):
		# For each neuron in k-th layer
		for i, neuron in enumerate(layer):
			# For each i-th layer connected with the neuron
			for key in neuron['weights']:
				#if neuron['shared']:
					# Do something
				#	pass 
				# If bias
				if key == 'bias':
					neuron['weights'][key] += alpha * neuron['epsilon']
				# Else weight
				else:
					# For each neuron in i-th layer potential connected with k-th layer
					for j in range(len(neuron['weights'][key])):
						if neuron['weights'][key][j] == NCN: continue
						if int(key) == 0:
							mu = (1.0 - momentum) * neuron['weights'][key][j] + momentum * inputs[j] * neuron['epsilon']
							neuron['weights'][key][j] += alpha * mu
						else:
							
							sum_share = 0.0
							# If the neuron is shared : Shared weights
							for share in shared:
								if (k, i, int(key), j) in share: 
									for couple in share:
										sum_share += network[k][i]['output'] * neuron['epsilon']
								if (int(key), j, k, i) in share:
									for couple in share:
										sum_share += network[k][i]['output'] * neuron['epsilon']
							#mu = (1.0 - momentum) * neuron['weights'][key][j] + momentum * network[int(key)-1][j]['output'] * neuron['epsilon']
							mu = (1.0 - momentum) * neuron['weights'][key][j] + momentum * sum_share
							neuron['weights'][key][j] += alpha * mu

# Cross entropy error function
def cross_entropy(excepted, predict):
	try:
		return - excepted * math.log((excepted / predict) + 10e-8, 2)
	except ValueError:
		print excepted, predict 

# MSE
def mean_squared(x, y):
	return (x - y) ** 2

# Decrease learning rate
def decay_learning_rate(alpha, tau, t):
	return alpha * tau / (tau + t) if tau > 0 else alpha

# Train the network
def train_network(network, trainset, shared, alpha, momentum, tau, num_iter, n_outputs):
	for epoch in range(num_iter):
		sum_error = 0
		current_alpha = decay_learning_rate(alpha, tau, epoch)
		#np.random.shuffle(trainset)
		for inputs in trainset:
			outputs = softmax(forward_propagate(network, [inputs[:-1]])[-1])
			expected = [0.0 for i in range(n_outputs)]
			expected[int(inputs[-1])-1] = 1.0
			#expected = [inputs[-1]]
			#sum_error += mean_squared(expected[-1], outputs[-1])
			sum_error += sum([cross_entropy(expected[i], outputs[i]) for i in range(n_outputs)])
			backward_propagate(network, expected)
			update_weights(network, inputs, current_alpha, momentum, shared)
		print "Epoch : {}; lrate = {:.3}; error={:.5}".format(epoch, current_alpha, sum_error)

# Compute softmax
def softmax(inputs):
	exp_in = map(math.exp, inputs)
	return [item / sum(exp_in) for item in exp_in]

# Make a prediction with a network
def predict(network, inputs):
	outputs = forward_propagate(network, inputs)
	return outputs[-1]
	#return outputs.index(max(outputs))

# Randomly uniform initialization
def initialize(network, shared):
	for layer in network:
		for neuron in layer:
			for connection in neuron['weights'].keys():
				if connection == 'bias':
					neuron[connection] = random.gauss(0, 1.0 / math.sqrt(len(layer)))
				else:
					for i in range(len(neuron['weights'][connection])):
						if neuron['weights'][connection][i] == NCN: continue
						neuron['weights'][connection][i] = random.gauss(0, 1.0 / math.sqrt(len(layer)))
	for share in shared:
		for t in share:
			network[t[0]][t[1]]['weights'][str(t[2])][t[3]] = network[share[0][0]][share[0][1]]['weights'][str(share[0][2])][share[0][3]]

random.seed(1)
dataset = load_file()
np.random.shuffle(dataset)

shared = [
	# Convonlutional C1
	[(0, 0, 0, 0), (0, 1, 0, 1), (0, 2, 0, 2)],
	[(0, 3, 0, 0), (0, 4, 0, 1), (0, 5, 0, 2)],
	[(0, 0, 0, 1), (0, 1, 0, 2), (0, 2, 0, 3)],
	[(0, 3, 0, 1), (0, 4, 0, 2), (0, 5, 0, 3)],

	# Conconlutional C2
	[(1, 0, 1, 0), (1, 1, 1, 1)],
	[(1, 0, 1, 1), (1, 1, 1, 2)],
	[(1, 0, 1, 3), (1, 1, 1, 4)],
	[(1, 0, 1, 4), (1, 1, 1, 5)],
	[(1, 2, 1, 0), (1, 3, 1, 1)],
	[(1, 2, 1, 1), (1, 3, 1, 2)],
	[(1, 2, 1, 3), (1, 3, 1, 4)],
	[(1, 2, 1, 4), (1, 3, 1, 5)],

	# Shared additional
	[(2, 0, 0, 0), (3, 0, 0, 0), (4, 0, 0, 0)],
	[(2, 0, 0, 1), (3, 0, 0, 1), (4, 0, 0, 1)],
	[(2, 0, 0, 2), (3, 0, 0, 2), (4, 0, 0, 2)],
	[(2, 0, 0, 3), (3, 0, 0, 3), (4, 0, 0, 3)],

	[(2, 1, 0, 0), (3, 1, 0, 0), (4, 1, 0, 0)],
	[(2, 1, 0, 1), (3, 1, 0, 1), (4, 1, 0, 1)],
	[(2, 1, 0, 2), (3, 1, 0, 2), (4, 1, 0, 2)],
	[(2, 1, 0, 3), (3, 1, 0, 3), (4, 1, 0, 3)],
]

network = load_network()

np.random.shuffle(dataset)
learning_rate = 0.05
tau = 50
momentum = 1.0
n_outputs = 3
epochs = 100

pp = pprint.PrettyPrinter()
initialize(network, shared)

train_network(network, dataset, shared, learning_rate, momentum, tau, epochs, n_outputs)
#for layer in network:
#	print(layer)
print predict(network, [[1, 0, 0, 0, None]])
print predict(network, [[0, 1, 1, 0, None]])
print predict(network, [[1, 0, 0, 1, None]])
