


import tensorflow as tf
from tensorflow import keras
import datetime
from tensorflow.keras.callbacks import ReduceLROnPlateau


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if logs.get('mae') <3:
            print("Reached 3 MAE")
            self.model.stop_training = True



def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)



log_dir = "/content/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks_list = [

    tf.keras.callbacks.EarlyStopping(
        monitor = 'val_mae',
        patience = 10
    ),

    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.00001),

    tf.keras.callbacks.ModelCheckpoint(
        filepath = 'mymodel.h5',
        monitor = 'val_loss',
        save_best_only = True
    ),

    tf.keras.callbacks.LearningRateScheduler(
        scheduler
    ),

    MyCallback(),

    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]


