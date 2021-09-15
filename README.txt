# Create the arquitecture of the model
# ¿Cuantas hidden layer usamos? --> [5,3], subtipo 2, 3 y 4
# ¿Modelo de activación "relu"? --> relu esta bien, probar sigmoid o tang hip

    # w_init = tf.random_normal_initializer()
    # w = tf.Variable(initial_value=w_init(shape=(1, Ctes.LAYERS[i]), dtype="float32"), trainable=True)
    # b_init = tf.zeros_initializer()
    # b = tf.Variable(initial_value=b_init(shape=(Ctes.LAYERS[i],), dtype="float32"), trainable=True)

Initializers: https://keras.io/api/layers/initializers/

    # Compile the model
    # ¿Que optimizador y loss se usan? --> Adam

    # Training the model
    # ¿Usamos data de validación? --> Se podría utilizar pero ver el tiempo de proceso. Base de datos chica, no conviene.
    # 100 Epochs
