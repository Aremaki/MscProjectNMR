#!/usr/bin/env python
# coding: utf-8

# # Training of DL regularized models for Multi-regression on 10000 spectra with independent concentrations 


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


# ## I.1 Large independent dataset (10000 spectra)


large_train_file_paths = ['../data/tfrecords/Concentrations_data/Large_sample/train/data_{}.tfrecord'.format(i) 
                        for i in range(32)]
large_val_file_paths = ['../data/tfrecords/Concentrations_data/Large_sample/validation/data_{}.tfrecord'.format(i) 
                      for i in range(8)]

large_train_dataset = read_tfrecords_concentrations(large_train_file_paths, 64)

large_val_dataset = read_tfrecords_concentrations(large_val_file_paths, 64)

large_train_datasets_single = []
large_val_datasets_single = []
for k in range(48):
    large_train_single_file_paths = ['../data/tfrecords/Concentrations_data/Large_sample_single/metabolite_{}/train/data_{}.tfrecord'.format(k, i) 
                        for i in range(32)]
    large_val_single_file_paths = ['../data/tfrecords/Concentrations_data/Large_sample_single/metabolite_{}/validation/data_{}.tfrecord'.format(k, i) 
                      for i in range(8)]
    large_train_datasets_single.append(read_tfrecords_concentrations_single(large_train_single_file_paths, 64))

    large_val_datasets_single.append(read_tfrecords_concentrations_single(large_val_single_file_paths, 64))



# ---
# # II. Define Callbacks

# ## II.1 Checkpoints


ckpt_large_multi_reg = CheckpointCallback("../saved_models/concentrations/large_multi_reg")
ckpt_large_single_reg = [CheckpointCallback("../saved_models/concentrations/large_single_reg/metabolite_{}"
                                             .format(i)) for i in range(48)]


# ## II.2 Earlystopping


earlystopping_50 = tf.keras.callbacks.EarlyStopping(patience=50) #for large dataset


# ## II.3 Training logs

logs_large_multi = tf.keras.callbacks.CSVLogger("../saved_logs/concentrations/large_multi_reg.csv")
logs_large_single = [tf.keras.callbacks.CSVLogger("../saved_logs/concentrations/large_single_reg/metabolite_{}.csv"
                                             .format(i)) for i in range(48)]

# ---
# # III. Train Models

# ## III.1 Large correlated dataset

# ### III.1.a Define mutli-output MLP

large_multi_reg_mlp = get_regularised_bn_dropout_mutli_regressor_mlp(input_shape=(10000,), hidden_units=[4096, 512],
                                                                 l2_reg_coeff=0,
                                                                 dropout_rate=0.2,
                                                                 num_outputs=48)
large_multi_reg_mlp.summary()


# ### III.1.b Compile mutli-output MLP


large_multi_reg_mlp.compile(optimizer=tf.keras.optimizers.Adam(0.0008), loss="mse",
                             metrics=[tf.keras.metrics.RootMeanSquaredError()])


# ### III.1.c Fit mutli-output MLP

large_multi_reg_mlp.fit(large_train_dataset, epochs=1000, validation_data=large_val_dataset,
                        callbacks=[ckpt_large_multi_reg, logs_large_multi_reg, earlystopping_100])

# ### III.1.e Define single-output MLPs


large_inde_reg_mlps = []

for i in range(48):
    large_inde_reg_mlps.append(get_regularised_bn_dropout_mutli_regressor_mlp(input_shape=(10000,),
                                                                          hidden_units=[4096, 512],
                                                                          l2_reg_coeff=0,
                                                                          dropout_rate=0.2,
                                                                          num_outputs=1))

large_inde_reg_mlps[0].summary()


# ### III.1.f Compile single-output MLPs


for i in range(48):
    large_inde_reg_mlp = large_inde_reg_mlps[i]
    large_inde_reg_mlp.compile(optimizer=tf.keras.optimizers.Adam(0.0008), loss="mse",
                        metrics=[tf.keras.metrics.RootMeanSquaredError()])


# ### III.1.g Fit single-output MLPs


for i in range(48):
    large_inde_reg_mlp = large_inde_reg_mlps[i]
    large_inde_reg_mlp.fit(large_train_datasets_single[i], epochs=1000,
                       validation_data=large_val_datasets_single[i],
                       callbacks=[ckpt_large_single_reg[i], logs_large_single_reg[i], earlystopping_100])

# ### III.1.h Reset GPU memory

device = cuda.get_current_device()
device.reset()