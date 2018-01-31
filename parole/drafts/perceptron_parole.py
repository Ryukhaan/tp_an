import random as random
import math as math
import numpy as np
import csv as csv


class Network():
	def __init__(self):
		self.layers = []

	def addLayerOfSize(self, nbNeurons):
			self.layers.append(Layer(nbNeurons))
	def init(self,input_size):
		nbL = len(self.layers)
		for l in range(nbL):
			if( ( l+1) == nbL):
				for n in self.layers[l].neurons:
					n.randomizeWeights(input_size)
			else:
				sourceLayerSize = len(self.layers[l+1].neurons)
				for n in self.layers[l].neurons:
					n.randomizeWeights(sourceLayerSize)
	def updateActivites(self, sourceInput):
		nbLayers=len(self.layers)
		self.layers[nbLayers-1].computeActivies(sourceInput)
		nbLayers = nbLayers - 1
		while(nbLayers > 0):
			print("another layer")
			self.layers[nbLayers-1].computeActivies(self.layers[nbLayers].neuronActivities)
			nbLayers = nbLayers - 1

class Layer():

	def __init__(self, nbNeurons):
		self.neurons = []  #list of neurons
		self.neuronActivities= [] #activities of each neuron of the layer
		for i in range(nbNeurons):
			self.neurons.append(Neuron())
			self.neuronActivities.append(0)
	def computeActivies(self, sourceLayer):
		for n in range(len(self.neurons)):
			s = 0
			for i in range(len(sourceLayer)):
				s = s + sourceLayer[i] * self.neurons[n].inputWeights[i]
			print(s)
			self.neuronActivities[n]= transfer(s, "Tanh")

class Neuron():
	def __init__(self):
		self.activity =0
		self.inputWeights = []

	def randomizeWeights(self, input_size):
		for w in range(input_size):
			self.inputWeights.append(random.random())






# Load file
def load_file():
	X = list()
	with open('parole.dt', 'r') as file :
		spamreader = csv.reader(file, delimiter=' ', quotechar='|')
		for row in spamreader:
			if not row: continue
			tmp = []
			for elem in row[0:241]:
				tmp.append(float(elem))
			X.append(tmp)
			#y.append(row[-1])
	return np.asarray(X)


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








random.seed(1)
data =load_file()
np.random.shuffle(data)

MLP = Network();

MLP.addLayerOfSize(1)
MLP.addLayerOfSize(1)
#MLP.addLayerOfSize(300)

input= [0,1,1]

MLP.init(len(input))

MLP.updateActivites(input)

print(input)

print(MLP.layers[1].neurons[0].inputWeights)
print(MLP.layers[1].neuronActivities)

print(MLP.layers[0].neurons[0].inputWeights)
print(MLP.layers[0].neuronActivities)




learning_rate = 0.2
tau = 200
