"""
bla bla.



bla.
"""

#Adding libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

from IPython.display import clear_output

# User Libraries
import Functions        # Funciones para crear la matriz y otras
import Graphs           # Funciones para graficar
import Ctes             # Constantes usadas
import Metrics_error    # Metricas de error
import Callbacks        # Callbacks utilizados en el aprendizaje
import Models           # Modelos de preodicción


def main():
    """bla."""
    path='/home/alejandro/Master Energías Renovables/Tesis Ale/4 - Dataset de prueba/'
    df = pd.read_excel(path+'datasetTestThesis2.xlsx')
    print(df.head())

    # Create matrix with given parameters
    mat = Functions.createMatrix(columnName=Ctes.COLUMN_NAME, data=df, regressorVariables=Ctes.REGRESSOR_VARIABLES, timestamp_K=Ctes.TIMESTAMP_K, timeLag=Ctes.TIME_LAG) # Funciona OK!

    # Convert matrix into a dataframe
    df_final = pd.DataFrame(mat)

    #Prepare training data
    x_train, y_train = Functions.getTrainData(df_final) # Divido las columnas del df en X e Y train
    print("{}\n{}".format(x_train.head(),y_train.head()))

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=Ctes.TEST_SIZE,random_state=Ctes.RANDOM_STATE)

    # Transform Pandas objects to TensorFlow object
    xnorm = StandardScaler()
    ynorm = StandardScaler()
    x_train=xnorm.fit_transform(x_train)
    x_test=xnorm.fit_transform(x_test)
    y_train=ynorm.fit_transform(np.array(y_train).reshape(-1,1))
    y_test=ynorm.fit_transform(np.array(y_test).reshape(-1,1))

    for i in range(Ctes.LOOPS):
        # Create the arquitecture of the model
        mod = Models.Model() # Creo un objeto del tipo Model para inicializar un nuevo modelo
        model = mod.model

        # Compile the model
        model.compile(loss=Ctes.LOSS, optimizer=tf.keras.optimizers.RMSprop(lr=Ctes.LEARNING_RATE), metrics=Ctes.METRICS)

        # Training the model
        history = model.fit(x_train,y_train,epochs=Ctes.EPOCHS,batch_size=Ctes.BATCH_SIZE,callbacks=[Callbacks.StopAtMinLoss(patience=Ctes.PATIENCE)], verbose=0) #, validation_split=0.2)

    # Recupero el mejor model
    with open('model_config.json') as json_file:
        json_config = json_file.read()
    best_model = keras.models.model_from_json(json_config)
    best_model.load_weights('best_weights.h5')

    # Prediction
    trainPredict = best_model.predict(x_train)
    testPredict = best_model.predict(x_test)

    # Inverse transform (Inverse standarization)
    y_PRED = ynorm.inverse_transform(testPredict)
    y_TEST = ynorm.inverse_transform(y_test)

    # Graphs
    plots = Graphs.Plots() # Creo un objeto de la clase Plots
    plots.PlotAll(y_TEST,y_PRED, history)

    # Metrics
    metrics = Metrics_error.Metricas()

    error = metrics.error(y_PRED,y_TEST)
    mae = metrics.mae(y_PRED,y_TEST)
    mse = metrics.mse(y_PRED,y_TEST)
    rmse = metrics.rmse(y_PRED,y_TEST)
    #metrics.mape(y_PRED,y_TEST)

    print("Error =\n",error)
    print("MAE =\n",mae)
    print("MSE =\n",mse)
    print("RMSE =\n",rmse)




if __name__ == "__main__":
    main()
