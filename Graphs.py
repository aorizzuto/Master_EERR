"""
bla bla.

bla.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math 

class Plots:
    """All Plots for this proyects will be in here."""

    def PlotHeatMap(self, data):
        """bla."""
        sns.heatmap(data.corr(),annot=True)

    def PlotAllDataActualVsPred(self, y_train,y_test,testPredict,show=True):
        """bla."""
        # y_train = what should I get
        plt.figure(figsize=(20,5))
        plt.plot((range(0,y_train.shape[0])),y_train, label='y_train')
        plt.plot(range(y_train.shape[0],y_train.shape[0]+y_test.shape[0]),y_test, label='y_test')
        plt.xlabel('Day')
        plt.ylabel('Mean Speed')
        plt.title('Wind Speed Prediction')
        plt.legend()

        # testPredict = what I get
        plt.figure(figsize=(20,5))
        plt.plot(range(0,y_train.shape[0]),y_train, label='y_train')
        plt.plot(range(y_train.shape[0],y_train.shape[0]+y_test.shape[0]),testPredict, label='testPredict')
        plt.xlabel('Day')
        plt.ylabel('Mean Speed')
        plt.title('Wind Speed Prediction')
        plt.legend()
        plt.show()

    def PlotActualVsPred(self, y_test,testPredict,rang=150,Title='Wind Speed Prediction',show=True):
        """bla."""
        # Plotting results
        print("PlotActualVsPred")

        plt.figure(figsize=(20,5))
        plt.plot(y_test[:rang], label='Actual')
        plt.plot(testPredict[:rang], label='Prediction')
        plt.xlabel('Hour')
        plt.ylabel('Mean Speed')
        plt.title(Title)
        plt.legend()
        plt.show()

    def PlotScatter(self, y_test,testPredict,show=True):
        """bla."""
        g=plt.scatter(y_test, testPredict)
        g.axes.set_xlabel('True Values ')
        g.axes.set_ylabel('Predictions ')
        g.axes.axis('equal')
        g.axes.axis('square')
        plt.show()

    def PlotDifference(self, y_test,testPredict,show=True):
        """bla."""
        plt.figure(figsize=(20,5))
        plt.plot(y_test - testPredict,marker='.',linestyle='')
        plt.show()

    def PlotLearningCurve(self, history,show=True):
        """bla."""
        plt.figure(figsize=(12,8))
        plt.plot(history.history['loss'])
        #plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Loss'])
        plt.show()

    def PlotAll(self, y_test, testPredict, history, Title='Wind Speed Prediction',rango=150):
        """bla."""
        grid = plt.GridSpec(2, 4)

        
        plt.figure(figsize=(25,7))

        sns.set_style("dark", {'axes.grid' : True})

        
        # First plot
        plt.subplot(grid[0,:])
        plt.plot(y_test[:rango], color='tab:blue', label='Actual')
        plt.plot(testPredict[:rango], 'r.',label='Prediction')
        plt.xlabel('Hour')
        plt.ylabel('Mean Speed')
        plt.title(Title)
        plt.legend(loc='lower left')


        # Second plot
        plt.subplot(grid[1, 0])
        g=plt.scatter(y_test, testPredict)
        g.axes.set_xlabel('True Values ')
        g.axes.set_ylabel('Predictions ')
        x = [l[0] for l in y_test]
        y = [l[0] for l in testPredict]
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x,p(x),"r--")

        # Third plot
        plt.subplot(grid[1, 2:])
        plt.plot(history.history['loss'])
        #plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Fourth plot
        plt.subplot(grid[1, 1])
        plt.plot(history.history['mse'])
        #plt.plot(history.history['val_loss'])
        plt.title('MSE')
        plt.xlabel('Epochs')
        plt.ylabel('mse')

        plt.tight_layout(3)
        plt.show()
        