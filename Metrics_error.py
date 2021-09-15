"""
bla bla.

bla.
"""

import numpy as np

class Metricas():
    """Metricas de error."""

    def error(self, y_predict, y_actual):
        """Cálculo de error entre predicción y actual."""
        return y_predict - y_actual

    def mse(self, y_predict, y_actual):
        """Cálculo de 'Mean Square Error' entre predicción y actual."""
        return np.square(y_predict - y_actual).mean()

    def rmse(self, y_predict, y_actual):
        """Cálculo de 'Root Mean Square Error' entre predicción y actual."""
        return np.sqrt(self.mse(y_predict, y_actual))

    def mae(self, y_predict, y_actual):
        """Cálculo de 'Mean Absolute Error' entre predicción y actual."""
        return np.abs(y_predict - y_actual).mean()

    def mape(self, y_predict, y_actual, x_valid):
        """Cálculo de 'Mean Absolute Percentage Error' entre predicción y actual."""
        return np.abs((y_predict - y_actual) / x_valid).mean()
