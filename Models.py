"""
bla.
"""

import Ctes
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten



class Model:
    """Creación de modelos de predicción."""

    model = None

    def __init__(self):
        """Inicialización del objeto Model."""
        
        self.createModel(layers=Ctes.LAYERS, activation_func=Ctes.ACT_FUNC)

    def createModel(self, layers, activation_func):
        self.model = Sequential()
        for i in range(len(layers)):
            kernelInitializer = initializers.RandomNormal(stddev=1.0, seed=None)
            biasInitializer = initializers.Zeros()
            self.model.add(Dense(units=layers[i], kernel_initializer=kernelInitializer, bias_initializer=biasInitializer, activation=activation_func))
        self.model.add(Dense(1))
