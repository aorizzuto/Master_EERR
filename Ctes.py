"""
bla bla.

bla.
"""

LOOPS               = 1

COLUMN_NAME         = 'wind speed'
REGRESSOR_VARIABLES = 5
TIMESTAMP_K         = 1
TIME_LAG            = 1



# Model
LAYERS              = [5,3] #  + 1 Output
ACT_FUNC            = 'sigmoid'
TEST_SIZE           = 0.3
RANDOM_STATE        = 101

# Compilation
OPTIMIZER           = 'adam'
LOSS                = 'mse'
METRICS             = ['mse', 'mae', 'mape', 'cosine_proximity']

# Fit
EPOCHS              = 300
BATCH_SIZE          = 32
PATIENCE            = 2 # 10
LEARNING_RATE       = 0.05 # 0.01