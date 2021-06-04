import tensorflow as tf


def get_simple_mlp(input_shape, hidden_units):
    """
    This function is used to build the simple MLP model. It takes input_shape and hidden_units
    as arguments, which should be used to build the model as described above, using the
    functional API. It returns the model
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    h = inputs
    for units in hidden_units:
        h = tf.keras.layers.Dense(units, activation='selu')(h)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(h)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def get_regularised_bn_dropout_mlp(input_shape, hidden_units, l2_reg_coeff, dropout_rate):
    """
    This function is used to build a complex MLP model. It takes input_shape, hidden_units,
    l2 regularised coefficient and dropout rate as arguments, which should be used to build
    the model as described above, using the functional API. It returns a model
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    h = inputs
    for units in hidden_units:
        h = tf.keras.layers.Dense(units, activation='selu',
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg_coeff))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.Dropout(dropout_rate)(h)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(h)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model
