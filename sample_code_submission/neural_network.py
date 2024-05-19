import numpy as np
import pandas as pd


class NeuralNetwork:
    """
    This Dummy class implements a neural network classifier
    change the code in the fit method to implement a neural network classifier


    """

    def __init__(self):
        self.model = None

    def fit(self,train_data):
        # Implement your fitting logic here
        pass

    def predict(self, test_data):
        # Implement your prediction logic here
        return np.random.randint(0, 1, len(test_data))
