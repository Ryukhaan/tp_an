import random as random
import math as math
import numpy as np 
import csv as csv
import json
import matplotlib.pyplot as plt

NCN = float('Inf')

# Load network.json
def load_network():
	with open('./network.json', 'rb'):
		alls = json.load(open('./network.json', 'rb'))
		network = alls['network']
		shared = alls['shared']
		for k, layer in enumerate(network):
			for i, neuron in enumerate(layer):
				for key in neuron['weights'].keys():
					if key == 'bias': continue
					for j in range(len(neuron['weights'][key])):
						if neuron['weights'][key][j] == "NCN":
							neuron['weights'][key][j] = NCN
				if neuron['function'] == "None":
					neuron['function'] = None
		for i, row in enumerate(shared):
			for j, link in enumerate(row):
				shared[i][j] = tuple(int(i) for i in str(link)[1:-1].split(','))
	return network, shared

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
	dataset_split = []
	dataset_copy = dataset
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		dataset_split.append(dataset_copy[i*fold_size:(i+1)*fold_size])
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
				# If bias
				if key == 'bias':
					#neuron['weights'][key] = neuron['weights'][key] - alpha * neuron['epsilon']
					neuron['weights'][key] += alpha * neuron['epsilon']
				# Else weight
				else:
					# For each neuron in i-th layer potentially connected with k-th layer
					for j in range(len(neuron['weights'][key])):
						# If there is no connection continue
						if neuron['weights'][key][j] == NCN: continue
						# If it's the output layer
						if int(key) == 0:
							mu = momentum * neuron['weights'][key][j] + inputs[j] * neuron['epsilon']
							#neuron['weights'][key][j] = neuron['weights'][key][j] - alpha * mu
							neuron['weights'][key][j] += alpha * mu
						else:
							sum_share = 0.0
							# If the neuron is shared : Shared weights
							if 'shared' in neuron['weights']:
								for share in shared:
									if (k, i, int(key), j) in share: 
										for couple in share:
											sum_share += network[k][i]['output'] * neuron['epsilon']
									elif (int(key), j, k, i) in share:
										for couple in share:
											sum_share += network[k][i]['output'] * neuron['epsilon']
							#mu = (1.0 - momentum) * neuron['weights'][key][j] + momentum * network[int(key)-1][j]['output'] * neuron['epsilon']
							mu = momentum * neuron['weights'][key][j] +  sum_share
							#neuron['weights'][key][j] = neuron['weights'][key][j] - alpha * mu
							neuron['weights'][key][j] += alpha * mu

# Cross entropy error function
def cross_entropy(excepted, predict):
	return - excepted * math.log(predict, 10)

# MSE
def mean_squared(x, y):
	return (x - y) ** 2

# Decrease learning rate
def decay_learning_rate(alpha, tau, t):
	return alpha * tau / (tau + t) if tau > 0 else alpha

# Train the network
def train(network, trainset, validset, testset, shared, alpha, momentum, tau, num_iter, n_outputs):
	list_valid_error = list()
	list_train_error = list()
	list_test_error = list()

	mean_valid_error = float('Inf')
	valid_error = 999.0
	epoch = 0

	# Early Stopping
	# We don't use mean because it seems that validation error can decrease for a long time (over 10000 iterations)
	while float('{0:.5}'.format(valid_error)) < float('{0:.5}'.format(mean_valid_error)) and epoch < num_iter:
		
		# Update value for error and early stopping
		mean_valid_error = valid_error
		train_error = 0.0
		valid_error = 0.0
		test_error = 0.0

		# Decay Learning Rate
		current_alpha = decay_learning_rate(alpha, tau, epoch)

		# Stochastic Gradient Descent
		# For each traning set example do
		for inputs in trainset:
			# Estimate outputs
			outputs = softmax(forward_propagate(network, [inputs[:-1]])[-1])

			# Create expected outputs
			expected = [0.0 for i in range(n_outputs)]
			expected[int(inputs[-1])-1] = 1.0

			# Sum error according to cross entropy error
			train_error += sum([cross_entropy(expected[i], outputs[i]) for i in range(n_outputs)])
			#print cross_entropy(expected[0], outputs[0]), expected[0], outputs[0]

			# Do backward propagation algorithm
			backward_propagate(network, expected)

			# Update weights and biases
			update_weights(network, inputs, current_alpha, momentum, shared)

		# Compute Error on validation dataset
		for inputs in validset:
			outputs = softmax(forward_propagate(network, [inputs[:-1]])[-1])
			expected = [0.0 for i in range(n_outputs)]
			expected[int(inputs[-1])-1] = 1.0
			valid_error += sum([cross_entropy(expected[i], outputs[i]) for i in range(n_outputs)])

		# Compute Error on test dataset
		for inputs in validset:
			outputs = softmax(forward_propagate(network, [inputs[:-1]])[-1])
			expected = [0.0 for i in range(n_outputs)]
			expected[int(inputs[-1])-1] = 1.0
			test_error += sum([cross_entropy(expected[i], outputs[i]) for i in range(n_outputs)])

		list_valid_error.append(valid_error)
		list_test_error.append(test_error)
		list_train_error.append(train_error)
		epoch += 1

	return list_train_error, list_valid_error, list_test_error

# Compute softmax
def softmax(inputs):
	m = max(inputs)
	# Prevents exp to overflow
	exp_in = map(math.exp, map(lambda x : x-m, inputs))
	return [item / sum(exp_in) for item in exp_in]

# Make a prediction with a network
def predict(network, inputs):
	outputs = forward_propagate(network, inputs)
	return outputs[-1]

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

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	# Create n-folds
	folds = cross_validation_split(dataset, n_folds)
	scores = list()

	# For each
	for i in range(len(folds)):
		testset = folds[i]
		score = list()
		for j in range(len(folds)):
			if i == j: continue
			validset = folds[j]

			trainset = [folds[x] for x in range(n_folds) if x!=j and x!=i]

			actual_trainset = []
			for n in range(len(trainset)):
				for row in trainset[n]:
					actual_trainset.append(row)

			# Train & Test
			predicted = algorithm(actual_trainset, validset, testset, *args)
			actual = [row[-1] for row in testset]
			predicted = map(lambda x : x+1, map(np.argmax, predicted))
			accuracy = accuracy_metric(actual, predicted)
			score.append(accuracy)

		scores.append(score)
	return scores

# Backpropagation Algorithm With Stochastic Gradient Descent
def train_algorithm(trainset, validset, testset, alpha, momentum, tau, epochs, n_outputs):
	network, shared = load_network()
	initialize(network, shared)
	trains, valids, tests = train(network, trainset, validset, testset, shared, alpha, momentum, tau, epochs, n_outputs)

	#fig = plt.figure()
	#plt.ion()
	#ax = fig.add_subplot(111)
	#x = np.linspace(0, len(trains), num=len(trains))
	#trains = ax.scatter(x, trains, s=5, c='red', marker="+")
	#valids = ax.scatter(x, valids, s=5, c='green', marker="+")
	#tests = ax.scatter(x, tests, s=5, c='blue', marker="+")
	#ax.set_title('Erreur au cours du temps', fontsize=14)
	#ax.set_xlabel('Nombre iterations')
	#ax.set_ylabel('Erreur totale')
	#ax.legend([trains, valids, tests], ["Base apprentissage", "Base validation", "Base test"])
	#plt.plot()
	#plt.draw()
	#plt.pause(15)

	predictions = list()
	for row in testset:
		prediction = predict(network, [row])
		predictions.append(prediction)
	return(predictions)

if __name__ == '__main__':
	random.seed(1)
	dataset = load_file()
	np.random.shuffle(dataset)
	learning_rate = 0.3
	tau = 200
	momentum = 0.00
	n_outputs = 3
	epochs = 10000
	n_folds = 5

	scores = evaluate_algorithm(dataset, train_algorithm, n_folds, learning_rate, momentum, tau, epochs, n_outputs)
	print('Scores: %s' % scores)
	for x in scores:
		print 'Mean Accuracy : {0:4f}'.format(sum(x)/float(len(x)))