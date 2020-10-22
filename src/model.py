import tensorflow as tf


def compile_and_fit(model, window, MAX_EPOCHS, patience=4):
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

Dense = tf.keras.Sequential([
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
    ])

CNN = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=12,
                           activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
    ])

RNN = tf.keras.Sequential([
    tf.keras.layers.LSTM(256,return_sequences=True),
    tf.keras.layers.LSTM(128,return_sequences=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
    ])
