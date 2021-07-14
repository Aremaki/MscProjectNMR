#!/usr/bin/env python
# coding: utf-8

# # Training of DL models for Multi-regression on 1000 spectra with independent concentrations 

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


small_train_file_paths = ['../data/tfrecords/Concentrations_data/Small_sample/train/data_{}.tfrecord'.format(i)
                    for i in range(8)]
small_val_file_paths = ['../data/tfrecords/Concentrations_data/Small_sample/validation/data_{}.tfrecord'.format(i)
                    for i in range(2)]

small_train_dataset = read_tfrecords_concentrations(small_train_file_paths, 32)

small_val_dataset = read_tfrecords_concentrations(small_val_file_paths, 32)



# ## I.2 Extract each metabolite from small dataset (for independent quantification)


def extract_metabolite(x, y, metab_index):
    return(x, y[..., metab_index])

small_train_datasets_single = [small_train_dataset.map(lambda x, y: extract_metabolite(x, y, i)) for i in range(48)]
small_val_datasets_single = [small_val_dataset.map(lambda x, y: extract_metabolite(x, y, i)) for i in range(48)]


# ---
# # II. Define Callbacks

# ## II.1 Checkpoints

ckpt_small_multi = CheckpointCallback("../saved_models/concentrations/small_multi")
ckpt_small_single = [CheckpointCallback("../saved_models/concentrations/small_single/metabolite_{}"
                                        .format(i)) for i in range(48)]


# ## II.2 Earlystopping


earlystopping_100 = tf.keras.callbacks.EarlyStopping(patience=100) #for small dataset


# ## II.3 Training logs


logs_small_multi = tf.keras.callbacks.CSVLogger("../saved_logs/concentrations/small_multi.csv")
logs_small_single = [tf.keras.callbacks.CSVLogger("../saved_logs/concentrations/small_single/metabolite_{}.csv"
                                        .format(i)) for i in range(48)]


# ---
# # III. Train Models

# ## III.1 Small independent dataset

# ### III.1.a Define mutli-output MLP


small_multi_mlp = get_simple_mutli_regressor_mlp(input_shape=(10000,), hidden_units=[512], num_outputs=48)
small_multi_mlp.summary()


# ### III.1.b Compile mutli-output MLP


small_multi_mlp.compile(optimizer="Adam", loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])


# ### III.1.c Fit mutli-output MLP


small_multi_mlp.fit(small_train_dataset, epochs=1000,
                                          validation_data=small_val_dataset,
                                          callbacks=[ckpt_small_multi, logs_small_multi,
                                                     earlystopping_100])


# ### III.1.e Define single-output MLPs


small_inde_mlps = []

for i in range(48):
    small_inde_mlps.append(get_simple_mutli_regressor_mlp(input_shape=(10000,),
                                                              hidden_units=[512],
                                                              num_outputs=1))

small_inde_mlps[0].summary()


# ### III.1.f Compile single-output MLPs


for i in range(48):
    small_inde_mlp = small_inde_mlps[i]
    small_inde_mlp.compile(optimizer="Adam", loss="mse",
                           metrics=[tf.keras.metrics.RootMeanSquaredError()])


# ### III.1.g Fit single-output MLPs


for i in range(48):
    small_inde_mlp = small_inde_mlps[i]
    small_inde_mlp.fit(small_train_datasets_single[i], epochs=1000,
                       validation_data=small_val_datasets_single[i],
                       callbacks=[ckpt_small_single[i], logs_small_single[i],
                                                                    earlystopping_100])


# ### III.1.h Reset GPU memory

device = cuda.get_current_device()
device.reset()