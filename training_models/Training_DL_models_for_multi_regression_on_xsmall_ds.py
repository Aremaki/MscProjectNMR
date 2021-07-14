#!/usr/bin/env python
# coding: utf-8

# # Training of DL models for Multi-regression on 100 spectra 

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
from models import get_simple_mutli_regressor_mlp, get_regularised_bn_dropout_mutli_regressor_mlp, CheckpointCallback



# ---
# # I. Read the tf.Record files

# ## I.1 Small independent dataset (1000 spectra)


xsmall_train_file_paths = ['../data/tfrecords/Concentrations_data/Extra_small_sample/train/data_{}.tfrecord'
                           .format(i) for i in range(4)]
xsmall_val_file_paths = ['../data/tfrecords/Concentrations_data/Extra_small_sample/validation/data_{}.tfrecord'
                         .format(i) for i in range(1)]

xsmall_train_dataset = read_tfrecords_concentrations(xsmall_train_file_paths, 16)

xsmall_val_dataset = read_tfrecords_concentrations(xsmall_val_file_paths, 16)

xsmall_corr_train_file_paths = ['../data/tfrecords/Concentrations_data/Extra_small_correlated/train/data_{}.tfrecord'
                               .format(i) for i in range(4)]
xsmall_corr_val_file_paths = ['../data/tfrecords/Concentrations_data/Extra_small_correlated/validation/data_{}.tfrecord'
                             .format(i) for i in range(1)]

xsmall_corr_train_dataset = read_tfrecords_concentrations(xsmall_corr_train_file_paths, 16)

xsmall_corr_val_dataset = read_tfrecords_concentrations(xsmall_corr_val_file_paths, 16)


# ## I.2 Extract each metabolite from small dataset (for independent quantification)


def extract_metabolite(x, y, metab_index):
    return(x, y[..., metab_index])

xsmall_train_datasets_single = [xsmall_train_dataset.map(lambda x, y: extract_metabolite(x, y, i)) for i in range(48)]
xsmall_val_datasets_single = [xsmall_val_dataset.map(lambda x, y: extract_metabolite(x, y, i)) for i in range(48)]
xsmall_corr_train_datasets_single = [xsmall_corr_train_dataset.map(lambda x, y: extract_metabolite(x, y, i)) 
                                    for i in range(48)]
xsmall_corr_val_datasets_single = [xsmall_corr_val_dataset.map(lambda x, y: extract_metabolite(x, y, i)) 
                                   for i in range(48)]

# ---
# # II. Define Callbacks

# ## II.1 Checkpoints

ckpt_xsmall_multi = CheckpointCallback("../saved_models/concentrations/extra_small_multi")
ckpt_xsmall_corr_multi = CheckpointCallback("../saved_models/concentrations/extra_small_corr_multi")
ckpt_xsmall_single = [CheckpointCallback("../saved_models/concentrations/extra_small_single/metabolite_{}"
                                        .format(i)) for i in range(48)]
ckpt_xsmall_corr_single = [CheckpointCallback("../saved_models/concentrations/extra_small_corr_single/metabolite_{}"
                                             .format(i)) for i in range(48)]


# ## II.2 Earlystopping


earlystopping_100 = tf.keras.callbacks.EarlyStopping(patience=100) #for small dataset


# ## II.3 Training logs


logs_xsmall_multi = tf.keras.callbacks.CSVLogger("../saved_logs/concentrations/extra_small_multi.csv")
logs_xsmall_corr_multi = tf.keras.callbacks.CSVLogger("../saved_logs/concentrations/extra_small_corr_multi.csv")
logs_xsmall_single = [tf.keras.callbacks.CSVLogger("../saved_logs/concentrations/extra_small_single/metabolite_{}.csv"
                                        .format(i)) for i in range(48)]
logs_xsmall_corr_single = [tf.keras.callbacks.CSVLogger("../saved_logs/concentrations/extra_small_corr_single/metabolite_{}.csv"
                                             .format(i)) for i in range(48)]


# ---
# # III. Train Models

# ## III.1 Extra small independent dataset

# ### III.1.a Define mutli-output MLP


xsmall_multi_mlp = get_simple_mutli_regressor_mlp(input_shape=(10000,), hidden_units=[256], num_outputs=48)
xsmall_multi_mlp.summary()


# ### III.1.b Compile mutli-output MLP


xsmall_multi_mlp.compile(optimizer=tf.keras.optimizers.Adam(0.002), loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])

# ### III.1.c Fit mutli-output MLP


xsmall_multi_mlp.fit(xsmall_train_dataset, epochs=1000, validation_data=xsmall_val_dataset,
                     callbacks=[ckpt_xsmall_multi, logs_xsmall_multi, earlystopping_100])


# ### III.1.e Define single-output MLPs


xsmall_inde_mlps = []

for i in range(48):
    xsmall_inde_mlps.append(get_simple_mutli_regressor_mlp(input_shape=(10000,),
                                                              hidden_units=[256],
                                                              num_outputs=1))

xsmall_inde_mlps[0].summary()


# ### III.1.f Compile single-output MLPs


for i in range(48):
    xsmall_inde_mlp = xsmall_inde_mlps[i]
    xsmall_inde_mlp.compile(optimizer=tf.keras.optimizers.Adam(0.002), loss="mse",
                           metrics=[tf.keras.metrics.RootMeanSquaredError()])


# ### III.1.g Fit single-output MLPs


for i in range(48):
    xsmall_inde_mlp = xsmall_inde_mlps[i]
    xsmall_inde_mlp.fit(xsmall_train_datasets_single[i], epochs=1000,
                       validation_data=xsmall_val_datasets_single[i],
                       callbacks=[ckpt_xsmall_single[i], logs_xsmall_single[i],
                                                                    earlystopping_100])
    
# ## III.2 Extra small correlated dataset

# ### III.2.a Define mutli-output MLP


xsmall_corr_multi_mlp = get_simple_mutli_regressor_mlp(input_shape=(10000,), hidden_units=[256], num_outputs=48)
xsmall_corr_multi_mlp.summary()


# ### III.2.b Compile mutli-output MLP


xsmall_corr_multi_mlp.compile(optimizer=tf.keras.optimizers.Adam(0.0025), loss="mse",
                              metrics=[tf.keras.metrics.RootMeanSquaredError()])
# ### III.2.c Fit mutli-output MLP


xsmall_corr_multi_mlp.fit(xsmall_corr_train_dataset, epochs=1000, validation_data=xsmall_corr_val_dataset,
                          callbacks=[ckpt_xsmall_corr_multi, logs_xsmall_corr_multi, earlystopping_100])


# ### III.2.e Define single-output MLPs


xsmall_corr_inde_mlps = []

for i in range(48):
    xsmall_corr_inde_mlps.append(get_simple_mutli_regressor_mlp(input_shape=(10000,),
                                                              hidden_units=[256],
                                                              num_outputs=1))

xsmall_corr_inde_mlps[0].summary()


# ### III.2.f Compile single-output MLPs


for i in range(48):
    xsmall_corr_inde_mlp = xsmall_corr_inde_mlps[i]
    xsmall_corr_inde_mlp.compile(optimizer=tf.keras.optimizers.Adam(0.0025), loss="mse",
                           metrics=[tf.keras.metrics.RootMeanSquaredError()])


# ### III.2.g Fit single-output MLPs


for i in range(48):
    xsmall_corr_inde_mlp = xsmall_corr_inde_mlps[i]
    xsmall_corr_inde_mlp.fit(xsmall_corr_train_datasets_single[i], epochs=1000,
                       validation_data=xsmall_corr_val_datasets_single[i],
                       callbacks=[ckpt_xsmall_corr_single[i], logs_xsmall_corr_single[i], earlystopping_100])


# ### III.3 Reset GPU memory

device = cuda.get_current_device()
device.reset()