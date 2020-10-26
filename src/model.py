import tensorflow as tf


def compile_and_fit(model, window, MAX_EPOCHS, patience=4):
    '''
    Function to pass a model object,dataset and max number of 
    epochs. The function has a callback which will stop the 'fit'
    early once minimal variation in loss is achieved.
    Input: model - Tensorflow object
           window - Dataset to be analyzed
           MAX_EPOCHS - Limit the epochs
    Output: history object for specified model.
    '''
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=patience,
                                                  mode='min')

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=["mae"])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stop])

    return history

# Dense Neural Network Definition
Dense = tf.keras.Sequential([
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
    ])

# Conv1D Network Definition
CNN = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=12,
                           activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
    ])

# Recurrent Neural Netowrk Definition
RNN = tf.keras.Sequential([
    tf.keras.layers.LSTM(256,return_sequences=True),
    tf.keras.layers.LSTM(128,return_sequences=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
    ])
