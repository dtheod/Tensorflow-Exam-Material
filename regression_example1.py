import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations, layers
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np


raw_train_data = pd.read_csv("Data/mercedes-benz-greener-manufacturing/train.csv")
raw_test_data = pd.read_csv("Data/mercedes-benz-greener-manufacturing/test.csv")

new_df = raw_train_data.select_dtypes(include = 'object')
new_df_dum = pd.get_dummies(raw_train_data, columns = list(new_df.columns) , prefix_sep='_')
test_new_df_dum = pd.get_dummies(raw_train_data, columns = list(new_df.columns) , prefix_sep='_')

"""
Outlier Detection
"""

clf = IsolationForest(max_samples = 100, random_state = 32)
clf.fit(new_df_dum)
y_noano = clf.predict(new_df_dum)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])

y_noano[y_noano['Top'] == 1].index.values

train = new_df_dum.iloc[y_noano[y_noano['Top'] == 1].index.values]
train.reset_index(drop = True, inplace = True)
print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
print("Number of rows without outliers:", train.shape[0])


"""
Split features and labels
"""

train_data = train.sample(frac = 0.5, random_state = 0)
validation_data = train.drop(train_data.index)


train_features = train_data.drop(['ID', 'y'], axis = 1).reset_index(drop = True)
train_labels = train_data.pop('y').reset_index(drop = True)

val_features = validation_data.drop(['ID', 'y'], axis = 1).reset_index(drop = True)
val_labels = validation_data.pop('y').reset_index(drop = True)


normalizer = tf.keras.layers.Normalization(axis = -1)
normalizer.adapt(train_features)


model = keras.Sequential(
    [
        layers.Dense(64, activation = 'relu'),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(1)
    ]
)

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if logs.get('mae') < 6:
            print("--------------------")
            print("Reached val_mae < 13")
            print("--------------------")
            self.model.stop_training = True

callback_list = [
    MyCallback(),
    
    tf.keras.callbacks.ModelCheckpoint(
        filepath = 'checkpoint_regression.keras',
        monitor = 'val_loss',
        save_best_only = True
    )
]

model.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.001),
               loss = 'mse', metrics = 'mae')

model.fit(train_features, train_labels,
          epochs = 50, batch_size = 32,
          validation_data = (val_features, val_labels)
          )

#print(model.evaluate(val_features, val_labels))

























