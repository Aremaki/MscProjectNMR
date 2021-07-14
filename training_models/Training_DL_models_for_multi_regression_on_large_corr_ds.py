#!/usr/bin/env python
# coding: utf-8

# # Training of DL models for Multi-regression on 10000 spectra with correlated concentrations 


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from numba import cuda

import joblib
import os

from tfrecords import read_tfrecords_concentrations, read_tfrecords_concentrations_single
from models import get_simple_mutli_regressor_mlp, CheckpointCallback


# ---
# # I. Read the tf.Record files


# ## I.1 Large correlated dataset (10000 spectra)

large_corr_train_file_paths = ['../data/tfrecords/Concentrations_data/Large_correlated/train/data_{}.tfrecord'
                               .format(i) for i in range(32)]
large_corr_val_file_paths = ['../data/tfrecords/Concentrations_data/Large_correlated/validation/data_{}.tfrecord'
                             .format(i) for i in range(8)]

large_corr_train_dataset = read_tfrecords_concentrations(large_corr_train_file_paths, 64)

large_corr_val_dataset = read_tfrecords_concentrations(large_corr_val_file_paths, 64)

large_corr_train_datasets_single = []
large_corr_val_datasets_single = []
for k in range(48):
    large_corr_train_single_file_paths = ['../data/tfrecords/Concentrations_data/Large_corr_single/metabolite_{}/train/data_{}.tfrecord'.format(k, i) 
                        for i in range(32)]
    large_corr_val_single_file_paths = ['../data/tfrecords/Concentrations_data/Large_corr_single/metabolite_{}/validation/data_{}.tfrecord'.format(k, i) 
                      for i in range(8)]
    large_corr_train_datasets_single.append(read_tfrecords_concentrations_single(large_corr_train_single_file_paths, 64))

    large_corr_val_datasets_single.append(read_tfrecords_concentrations_single(large_corr_val_single_file_paths, 64))


# ---
# # II. Define Callbacks

# ## II.1 Checkpoints


ckpt_large_corr_multi = CheckpointCallback("../saved_models/concentrations/large_corr_multi")
ckpt_large_corr_single = [CheckpointCallback("../saved_models/concentrations/large_corr_single/metabolite_{}"
                                             .format(i)) for i in range(48)]


# ## II.2 Earlystopping


earlystopping_50 = tf.keras.callbacks.EarlyStopping(patience=50) #for large dataset


# ## II.3 Training logs

logs_large_corr_multi = tf.keras.callbacks.CSVLogger("../saved_logs/concentrations/large_corr_multi.csv")
logs_large_corr_single = [tf.keras.callbacks.CSVLogger("../saved_logs/concentrations/large_corr_single/metabolite_{}.csv"
                                             .format(i)) for i in range(48)]

# ---
# # III. Train Models

# ## III.1 Large correlated dataset

# ### III.1.a Define mutli-output MLP

large_corr_multi_mlp = get_simple_mutli_regressor_mlp(input_shape=(10000,), hidden_units=[256, 256], num_outputs=48)
large_corr_multi_mlp.summary()


# ### III.1.b Compile mutli-output MLP


large_corr_multi_mlp.compile(optimizer="Adam", loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])


# ### III.1.c Fit mutli-output MLP

large_corr_multi_mlp.fit(large_corr_train_dataset, epochs=1000,
                                          validation_data=large_corr_val_dataset,
                                          callbacks=[ckpt_large_corr_multi, logs_large_corr_multi,
                                                     earlystopping_100])

# ### III.1.e Define single-output MLPs


large_corr_inde_mlps = []

for i in range(48):
    large_corr_inde_mlps.append(get_simple_mutli_regressor_mlp(input_shape=(10000,), hidden_units=[256, 256], num_outputs=1))

large_corr_inde_mlps[0].summary()


# ### III.1.f Compile single-output MLPs


for i in range(48):
    large_corr_inde_mlp = large_corr_inde_mlps[i]
    large_corr_inde_mlp.compile(optimizer="Adam", loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])


# ### III.1.g Fit single-output MLPs


for i in range(48):
    large_corr_inde_mlp = large_corr_inde_mlps[i]
    large_corr_inde_mlp.fit(large_corr_train_datasets_single[i], epochs=1000,
                       validation_data=large_corr_val_datasets_single[i],
                       callbacks=[ckpt_large_corr_single[i], logs_large_corr_single[i], earlystopping_100])

# ### III.1.h Reset GPU memory

device = cuda.get_current_device()
device.reset()