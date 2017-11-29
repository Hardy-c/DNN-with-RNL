# RNL for Deep Learning

Random neural layers come from queue theory. Here, we propose to add RNL in classical deep neural network. We add RNL as a Keras layer class.

## List of files

 * RNL.py : Random neural layer class for Keras
 * cifar10_keras.py : Example with a CNN on Cifar10
 * mnist\_mlp\_keras.py : Example with a MLN on MNIST
 * mnist\_cnn\_keras.py : Example with a CNN on MNIST

## Getting Started

### Prerequisites

AdaComp requires :

* Python 2.7
* Tensorflow (>=1.0)
* Keras

### Running

To run one of example, execute :

	python _name_of_file.py_


### Add one or more RNL in your Keras 

	import keras
	from RNL import RNL
	
	model = Sequential()
	[...]
	model.add(RNL(nb_neurons))
	[...]


## Author

* Corentin Hardy
