# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 15:32:15 2025

@author: david
"""

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

name = "test2" #"adidas_vintage"
#Parquet Path
local_path_1 = r"C:\Users\david\OneDrive - fs-students.de\Vinted_Data\Data\\"+name+"_data_parquet"
# Path to your .h5 file
h5_path = r"C:\Users\david\OneDrive - fs-students.de\Vinted_Data\Data\\"+name+"_data_img.h5"


# Create a generator to yield batches from the HDF5 file
def h5_batch_generator(h5_path, batch_size):
    with h5py.File(h5_path, 'r') as f:
        X = f['images']   # shape e.g. (N, H, W, C)
        y = f['labels']   # shape e.g. (N,)
        n_samples = X.shape[0]
        idxs = np.arange(n_samples)
        
        while True:  # infinite generator for Keras fit()
            np.random.shuffle(idxs)
            for i in range(0, n_samples, batch_size):
                batch_idx = idxs[i:i+batch_size]
                X_batch = X[batch_idx].astype(np.float32) / 255.0  # simple normalization
                y_batch = y[batch_idx]
                yield X_batch, y_batch

# Example model
model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Get total number of samples (without loading all data)
with h5py.File(h5_path, 'r') as f:
    n_samples = f['images'].shape[0]

batch_size = 64
steps_per_epoch = n_samples // batch_size

# Train model using generator
gen = h5_batch_generator(h5_path, batch_size)
model.fit(gen, steps_per_epoch=steps_per_epoch, epochs=5)
