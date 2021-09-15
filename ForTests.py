import Functions, Graphs, Ctes, Metrics_error, Callbacks, Models
import os.path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras

x = [1,2,3,4,5,6,7]
y = [1,3,2,4,6,5,7]

sns.set_style("dark", {'axes.grid' : True})
sns.lineplot(x=x, y=range(len(x)))
sns.axes_style("dark")
plt.xlim((0,8))
plt.ylim((0,8))
plt.show()

