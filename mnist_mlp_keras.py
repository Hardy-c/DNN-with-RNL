'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from RNL import RNL
from keras import backend as K


batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


config = [
    ["Dense","Dense","Dense"],
    ["Dense","Dense","Random"],
    ["Dense","Random","Dense"],
    ["Dense","Random","Random"],
    ["Random","Dense","Dense"],
    ["Random","Dense","Random"],
    ["Random","Random","Dense"],
    ["Random","Random","Random"]
]
for i in range(5):
    for c in config[::-1]:
        model = Sequential()
        if c[0] == "Dense":
            model.add(Dense(512, activation='relu', input_shape=(784,)))
        elif c[0] == "Random":
            model.add(RNL(512, input_shape=(784,)))
        else:
            raise Exception("Error in config !")
        model.add(Dropout(0.2))
        
        if c[1] == "Dense":
            model.add(Dense(512, activation='relu'))
        elif c[1] == "Random":
            model.add(RNL(512))
        else:
            raise Exception("Error in config !")
        
        model.add(Dropout(0.2))

        if c[2] == "Dense":
            model.add(Dense(num_classes))
        elif c[2] == "Random":
            model.add(RNL(num_classes))
        else:
            raise Exception("Error in config !")
        
        model.add(Dropout(0.2))
        model.add(Activation(K.softmax))

        
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Configuration :',c)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
