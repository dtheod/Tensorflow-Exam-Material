import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
import pandas as pd
import numpy as np

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.dropna(inplace = True)
dataset['Origin'] = dataset['Origin'].map({1:'USA', 2:'Europe', 3:'Japan'})
dataset = pd.get_dummies(dataset, columns = ['Origin'], prefix='', prefix_sep='')

train_data = dataset.sample(frac = 0.8, random_state=0)
test_data = dataset.drop(train_data.index)


train_features = train_data.copy()
test_features = test_data.copy()

train_labels = train_data.pop('MPG')
test_labels = test_data.pop('MPG')


normalizer = tf.keras.layers.Normalization(axis = -1)
normalizer.adapt(np.array(train_features))


model = keras.Sequential(
    [
        normalizer,
        layers.Dense(64, activation = 'relu'),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(1)
    ]
)

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if logs.get('mae') < .9:
            print("Reached .9 MAE")
            self.model.stop_training = True

model.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.01), loss = 'mse', metrics = 'mae')


my_callback_list = [

    MyCallback(),
    
    tf.keras.callbacks.ModelCheckpoint(
        filepath = 'checkpoint_regression.keras',
        monitor = 'val_loss',
        save_best_only = True
    )
]


model.fit(train_features,
          train_labels,
          epochs = 20,
          validation_split = 0.2,
          callbacks = my_callback_list)


print("Evaluation")
print(model.evaluate(test_features, test_labels))












