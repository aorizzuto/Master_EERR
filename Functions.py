"""
bla bla.

bla.
"""

import pandas as pd

def createMatrix(columnName,data,regressorVariables,timestamp_K=1,timeLag=1):
    """Look for holes in "inputs" consecutive values. If we found a hole in those consecutive values, then the sublist will be ignored.
    
    Parameters:
        columnName: name of the column
        data: data to check
        regressorVariables: window of inputs
        timestamp_K: K target
        timeLag: Space between input data
    Return:
        lstToReturn: Matrix with sublists
        
    Example:
    
    X: value
    O: hole
    inputsTarget: 4
    
    0  1                        10
    |  |                        |
    [X0 X1 X2 X3 X4 O X5 X6 X7 O X8 X9 X10 X11]
    """  
    lstToReturn = []

    for position in range(data.shape[0]): # Cycling through dataset
        last_position = position + ((regressorVariables-1)*timeLag +1)
        try:
            listToCheck = data[columnName][position : last_position : timeLag]
            targetValue = data[columnName][last_position + timestamp_K - 1]
        
            lstFinal = listToCheck.tolist()
            lstFinal.append(targetValue)

            if not(any(pd.isna(lstFinal))):
                lstToReturn.append(lstFinal)
        except Exception as e:
            print('ERROR: ',e)
            break

    return lstToReturn 


def getTrainData (df_final):
    x_train = df_final.drop([df_final.columns.stop-1],axis=1)
    y_train = df_final[df_final.columns.stop-1]
    return x_train, y_train