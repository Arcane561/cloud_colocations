import numpy as np
import keras
from keras.models import Sequential, clone_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

class SimpleClassifier:

    def __init__(self, x, y, width):
        self.input_dim = np.prod(x.shape[1:])
        self.width = width
        self.n_classes = y.shape[1]

        self.model = Sequential()
        self.model.add(Dense(input_dim = self.input_dim,
                             units = self.width,
                             activation = "relu"))
        self.model.add(Dense(input_dim = self.width,
                    units = self.n_classes,
                    activation = 'softmax'))
        self.model.compile(loss = 'categorical_crossentropy',
                           optimizer = 'adam',
                           metrics = ['accuracy'])

class DeepClassifier:

    def __init__(self, x, y, width):
        self.input_dim = np.prod(x.shape[1:])
        self.width = width
        self.n_classes = y.shape[1]

        self.model = Sequential()
        self.model.add(Dense(input_dim = self.input_dim,
                             units = self.width,
                             activation = "relu"))
        self.model.add(Dense(input_dim = self.width,
                    units = self.width,
                    activation = 'relu'))
        self.model.add(Dense(input_dim = self.width,
                    units = self.n_classes,
                    activation = 'softmax'))
        self.model.compile(loss = 'categorical_crossentropy',
                           optimizer = 'adam',
                           metrics = ['accuracy'])


