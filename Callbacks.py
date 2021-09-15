"""
bla bla.

bla.
"""

import datetime
import tensorflow as tf
import numpy as np
import os.path

# class CallbacksPrints(tf.keras.callbacks.Callback):
#     """Callbacks."""

#     def on_train_batch_end(self, batch, logs=None):
#         print('Para el batch {}, la perdida (loss) es {:7.2f}.'.format(batch, logs['loss']))

#     def on_test_batch_end(self, batch, logs=None):
#         print('Para el  batch {}, la perdida (loss) es {:7.2f}.'.format(batch, logs['loss']))

#     def on_epoch_end(self, epoch, logs=None):
#         print('La perdida promedio para la epoch {} es {:7.2f}.'.format(epoch, logs['loss']))


class StopAtMinLoss(tf.keras.callbacks.Callback):
    """Detener el entrenamiento cuando la perdida (loss) esta en su minimo, i.e. la perdida (loss) deja de disminuir.

    Arguments:
        patience: Numero de epochs a esperar despues de que el min ha sido alcanzaado. Despues de este numero
        de no mejoras, el entrenamiento para.

    Interpreting get_weights. Weights depends of numbers of connections

        Weights for the first layer 
        Biases for the first layer 
        Weights for the second layer 
        Biases for the second layer 
    """

    def __init__(self, patience=0):
        super(StopAtMinLoss, self).__init__()

        self.patience = patience
        self.best_weights = None        # best_weights para almacenar los pesos en los cuales ocurre la perdida minima.

    def on_train_begin(self, logs=None):
    
        self.wait = 0                   # El numero de epoch que ha esperado cuando la perdida ya no es minima.
        self.stopped_epoch = 0          # El epoch en el que en entrenamiento se detiene.
        self.best = np.Inf              # Initialize el best como infinito.

    def on_epoch_end(self, epoch, logs=None):
        print('La perdida promedio para la epoch {} es {:4.6f}.'.format(epoch, logs['loss']))        
        current = logs.get('loss')
        if np.less(current, self.best):
            print("Se encontró un nuevo mínimo. Esperando {} epochs sin nuevos mínimos.\n".format(self.patience))
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights() # Guardar los mejores pesos si el resultado actual es mejor (menos).
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print('\nHan pasado {} epochs sin tener un nuevo mínimo.'.format(self.patience))
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Restaurando los pesos del modelo del final de la mejor epoch.\n')
                self.model.set_weights(self.best_weights)
                print('Los weights mínimos son:')
                w = self.best_weights
                for i in range(int(len(w)/2)):  
                    print('Layer {}:'.format(i))    
                    print('\tWeigths = ')
                    for j in range(len(w[i*2])):
                        print('\t\t',w[i*2][j])
                    print('\tBiases = ')
                    print('\t\t',w[i*2+1])
                self.check_loss(logs['loss'])

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('\nEpoch %05d: Detencion anticipada' % (self.stopped_epoch + 1))

    def check_loss(self, loss):
        try:
            file_obj = open("minLoss.txt","r")
            line = file_obj.readline().strip()
            file_obj.close()
            
            if float(line) > float(loss):
                with open("minLoss.txt", "a") as file_obj:  # Open a file with access mode 'a'
                    file_obj.seek(0)                        # absolute file positioning 
                    file_obj.truncate()                     # to erase all data 
                    file_obj.write(f"{loss}\n")             # Add weights to file
                    file_obj.close()
                
                # Guardar configuración JSON en el disco
                json_config = self.model.to_json()
                with open('model_config.json', 'a') as json_file:
                    json_file.seek(0)                       # Me paro al inicio
                    json_file.truncate()                    # Borro datos
                    json_file.write(json_config)
                
                self.model.save_weights('best_weights.h5')  # Guardar pesos en el disco
                print('**********************************')
                print('New best model! Loss = ',loss)
                print('**********************************\n\n')

        except:
            with open("minLoss.txt", "a") as file_obj:      # Open a file with access mode 'a'
                file_obj.write(f"{loss}\n")                 # Add weights to file
                file_obj.close()        

        
        


       

