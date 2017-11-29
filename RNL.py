from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform
import keras.constraints as contraints
import numpy as np

class RNL(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(RNL, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.wp = self.add_weight(name='positive_weights', 
                                  shape=(input_shape[1], self.output_dim),
                                  initializer=RandomUniform(minval=0, maxval=0.2),
                                  trainable=True,
                                  constraint=contraints.non_neg())
        self.wn = self.add_weight(name='negative_weights', 
                                  shape=(input_shape[1], self.output_dim),
                                  initializer=RandomUniform(minval=0, maxval=0.2),
                                  trainable=True,
                                  constraint=contraints.non_neg())
        super(RandomLayer, self).build(input_shape)

    def call(self, x):
        return K.dot(x, self.wp)/(K.dot(x, self.wn)+1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
