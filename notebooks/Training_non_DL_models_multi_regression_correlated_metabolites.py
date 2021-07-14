#!/usr/bin/env python
# coding: utf-8

# # Training of non-DL models for Multi-regression of the metabolites concentrations on correlated metabolites

# ### Install project packages

# In[4]:


get_ipython().run_cell_magic('bash', '', 'pip install -e ../.')


# ### Install required python modules

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'pip install -r ../requirements.txt')


# ### Import fucntions

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression

import os
import joblib

from tfrecords import read_tfrecords_concentrations, read_tfrecords_concentrations_single


# In[2]:


tf.__version__


# ---
# # I. Read the tf.Record files

# ## I.1 Small correlated dataset (1000 spectra)

# In[ ]:


small_corr_train_file_paths = ['../data/tfrecords/Concentrations_data/Small_correlated/train/data_{}.tfrecord'
                               .format(i) for i in range(8)]
small_corr_val_file_paths = ['../data/tfrecords/Concentrations_data/Small_correlated/validation/data_{}.tfrecord'
                             .format(i) for i in range(2)]

small_corr_train_dataset = read_tfrecords_concentrations(small_corr_train_file_paths, 32)

small_corr_val_dataset = read_tfrecords_concentrations(small_corr_val_file_paths, 32)


# In[ ]:


X_train_small_corr = []
y_train_small_corr = []

for element in small_corr_train_dataset.unbatch():
    X_train_small_corr.append(element[0].numpy())
    y_train_small.append(element[1].numpy())
    
for element in small_corr_val_dataset.unbatch():
    X_train_small_corr.append(element[0].numpy())
    y_train_small.append(element[1].numpy())

X_train_small_corr = pd.DataFrame(X_train_small_corr)
y_train_small_corr = pd.DataFrame(y_train_small_corr)


# ## I.2 Large correlated dataset (10000 spectra)

# In[ ]:


large_corr_train_file_paths = ['../data/tfrecords/Concentrations_data/Large_correlated/train/data_{}.tfrecord'
                               .format(i) for i in range(32)]
large_corr_val_file_paths = ['../data/tfrecords/Concentrations_data/Large_correlated/validation/data_{}.tfrecord'
                             .format(i) for i in range(8)]

large_corr_train_dataset = read_tfrecords_concentrations(large_corr_train_file_paths, 64)

large_corr_val_dataset = read_tfrecords_concentrations(large_corr_val_file_paths, 64)


# In[ ]:


X_train_large_corr = []
y_train_large_corr = []

for element in large_corr_train_dataset.unbatch():
    X_train_large_corr.append(element[0].numpy())
    y_train_large_corr.append(element[1].numpy())

for element in large_corr_val_dataset.unbatch():
    X_train_large_corr.append(element[0].numpy())
    y_train_large_corr.append(element[1].numpy())

X_train_large_corr = pd.DataFrame(X_train_large_corr)
y_train_large_corr = pd.DataFrame(y_train_large_corr)


# ## I.3 Extra small correlated dataset (100 spectra)

# In[ ]:


xsmall_corr_train_file_paths = ['../data/tfrecords/Concentrations_data/Extra_small_correlated/train/data_{}.tfrecord'
                               .format(i) for i in range(4)]
xsmall_corr_val_file_paths = ['../data/tfrecords/Concentrations_data/Extra_small_correlated/validation/data_{}.tfrecord'
                             .format(i) for i in range(1)]

xsmall_corr_train_dataset = read_tfrecords_concentrations(xsmall_corr_train_file_paths, 16)

xsmall_corr_val_dataset = read_tfrecords_concentrations(xsmall_corr_val_file_paths, 16)


# In[ ]:


X_train_xsmall_corr = []
y_train_xsmall_corr = []

for element in xsmall_corr_train_dataset.unbatch():
    X_train_xsmall_corr.append(element[0].numpy())
    y_train_xsmall_corr.append(element[1].numpy())
    
for element in xsmall_corr_val_dataset.unbatch():
    X_train_xsmall_corr.append(element[0].numpy())
    y_train_xsmall_corr.append(element[1].numpy())

X_train_xsmall_corr = pd.DataFrame(X_train_xsmall_corr)
y_train_xsmall_corr = pd.DataFrame(y_train_xsmall_corr)


# ---
# # II. Train Models

# ## II.1 Small correlated dataset

# ### II.1.a Define mutli-output Random Forest

# In[ ]:


small_corr_multi_RF = RandomForestRegressor(10)


# ### II.1.b Fit mutli-output Random Forest

# In[ ]:


if not os.path.exists("../saved_models/concentrations/small_corr_multi_RF"):
    os.makedirs("../saved_models/concentrations/small_corr_multi_RF")
start = time.time()
small_corr_multi_RF.fit(X_train_small_corr, y_train_small_corr)
time_small_corr = time.time() - start
joblib.dump(small_corr_multi_RF, "../saved_models/concentrations/small_corr_multi_RF/model")
joblib.dump(time_small_corr, "../saved_models/concentrations/small_corr_multi_RF/time")


# ### II.1.c Define single-output Random Forest

# In[24]:


small_corr_inde_RFs = []

for i in range(48):
    small_corr_inde_RFs.append(RandomForestRegressor(10))


# ### II.1.d Fit single-output Random Forest

# In[ ]:


for i in range(48):
    start = time.time()
    small_corr_inde_RFs[i].fit(X_train_small_corr, y_train_small_corr[i])
    time_small_corr_inde = time.time() - start
    if not os.path.exists("../saved_models/concentrations/small_corr_single_RF/metabolite_{}".format(i)):
        os.makedirs("../saved_models/concentrations/small_corr_single_RF/metabolite_{}".format(i)):
    joblib.dump(small_corr_inde_RFs[i], "../saved_models/concentrations/small_corr_single_RF/metabolite_{}/model"
                .format(i))
    joblib.dump(time_small_corr_inde, "../saved_models/concentrations/small_corr_single_RF/metabolite_{}/time"
                .format(i))
    print('\n', '\n', '####', '\n', '\n', 'Model number {} has been trained in {} seconds !'
          .format(i, time_small_corr_inde), '\n', '\n', '####', '\n', '\n')


# ### II.1.e Define partial least squares

# In[27]:


small_corr_pls = PLSRegression(200)


# ### II.1.f Fit partial least squares

# In[ ]:


if not os.path.exists("../saved_models/concentrations/small_corr_pls"):
    os.makedirs("../saved_models/concentrations/small_corr_pls")
start = time.time()
small_corr_pls.fit(X_train_small_corr, y_train_small_corr)
time_small_corr = time.time() - start
joblib.dump(small_corr_pls, "../saved_models/concentrations/small_corr_pls/model")
joblib.dump(time_small_corr, "../saved_models/concentrations/small_corr_pls/time")


# ## II.2 Large correlated dataset

# ### II.2.a Define mutli-output Random Forest

# In[36]:


large_corr_multi_RF = RandomForestRegressor(10)


# ### II.2.b Fit mutli-output Random Forest

# In[ ]:


if not os.path.exists("../saved_models/concentrations/large_corr_multi_RF"):
    os.makedirs("../saved_models/concentrations/large_corr_multi_RF")
start = time.time()
large_corr_multi_RF.fit(X_train_large_corr, y_train_large_corr)
time_large_corr = time.time() - start
joblib.dump(large_corr_multi_RF, "../saved_models/concentrations/large_corr_multi_RF/model")
joblib.dump(time_large_corr, "../saved_models/concentrations/large_corr_multi_RF/time")


# ### II.2.c Define single-output Random Forest

# In[24]:


large_corr_inde_RFs = []

for i in range(48):
    large_corr_inde_RFs.append(RandomForestRegressor(10))


# ### II.2.d Fit single-output Random Forest

# In[ ]:


if not os.path.exists("../saved_models/concentrations/large_corr_single_RF"):
    os.makedirs("../saved_models/concentrations/large_corr_single_RF")
for i in range(48):
    start = time.time()
    large_corr_inde_RFs[i].fit(X_train_large_corr, y_train_large_corr[i])
    time_large_corr_inde = time.time() - start
    if not os.path.exists("../saved_models/concentrations/large_corr_single_RF/metabolite_{}".format(i)):
        os.makedirs("../saved_models/concentrations/large_corr_single_RF/metabolite_{}".format(i)):
    joblib.dump(large_corr_inde_RFs[i], "../saved_models/concentrations/large_corr_single_RF/metabolite_{}/model"
                .format(i))
    joblib.dump(time_large_corr_inde, "../saved_models/concentrations/large_corr_single_RF/metabolite_{}/time"
                .format(i))
    print('\n', '\n', '####', '\n', '\n', 'Model number {} has been trained in {} seconds !'
          .format(i, time_large_corr_inde), '\n', '\n', '####', '\n', '\n')


# ### II.2.e Define partial least squares

# In[27]:


large_corr_pls = PLSRegression(200)


# ### II.2.f Fit partial least squares

# In[ ]:


if not os.path.exists("../saved_models/concentrations/large_corr_pls"):
    os.makedirs("../saved_models/concentrations/large_corr_pls")
start = time.time()
large_corr_pls.fit(X_train_large_corr, y_train_large_corr)
time_large_corr = time.time() - start
joblib.dump(large_corr_pls, "../saved_models/concentrations/large_corr_pls/time")
joblib.dump(time_large_corr, "../saved_models/concentrations/large_corr_pls/time")


# ## II.3 Extra small correlated dataset

# ### II.3.a Define mutli-output Random Forest

# In[50]:


xsmall_corr_multi_RF = RandomForestRegressor(10)


# ### II.3.b Fit mutli-output Random Forest

# In[ ]:


if not os.path.exists("../saved_models/concentrations/xsmall_corr_multi_RF"):
    os.makedirs("../saved_models/concentrations/xsmall_corr_multi_RF")
start = time.time()
xsmall_corr_multi_RF.fit(X_train_small_corr, y_train_small_corr)
time_xsmall_corr = time.time() - start
joblib.dump(xsmall_corr_multi_RF, "../saved_models/concentrations/xsmall_corr_multi_RF/model")
joblib.dump(time_xsmall_corr, "../saved_models/concentrations/xsmall_corr_multi_RF/time")


# ### II.3.c Define single-output Random Forest

# In[24]:


xsmall_corr_inde_RFs = []

for i in range(48):
    xsmall_corr_inde_RFs.append(RandomForestRegressor(10))


# ### II.3.d Fit single-output Random Forest

# In[ ]:


for i in range(48):
    start = time.time()
    xsmall_corr_inde_RFs[i].fit(X_train_xsmall_corr, y_train_xsmall_corr[i])
    time_xsmall_corr_inde = time.time() - start
    if not os.path.exists("../saved_models/concentrations/xsmall_corr_single_RF/metabolite_{}".format(i)):
        os.makedirs("../saved_models/concentrations/xsmall_corr_single_RF/metabolite_{}".format(i)):
    joblib.dump(xsmall_corr_inde_RFs[i], "../saved_models/concentrations/xsmall_corr_single_RF/metabolite_{}/model"
                .format(i))
    joblib.dump(time_xsmall_corr_inde, "../saved_models/concentrations/xsmall_corr_single_RF/metabolite_{}/time"
                .format(i))
    print('\n', '\n', '####', '\n', '\n', 'Model number {} has been trained in {} seconds !'
          .format(i, time_xsmall_corr_inde), '\n', '\n', '####', '\n', '\n')


# ### II.3.e Define partial least squares

# In[27]:


xsmall_corr_pls = PLSRegression(200)


# ### II.3.f Fit partial least squares

# In[ ]:


if not os.path.exists("../saved_models/concentrations/xsmall_corr_pls"):
    os.makedirs("../saved_models/concentrations/xsmall_corr_pls")
start = time.time()
xsmall_corr_pls.fit(X_train_xsmall_corr, y_train_xsmall_corr)
time_xsmall_corr = time.time() - start
joblib.dump(xsmall_corr_pls, "../saved_models/concentrations/xsmall_corr_pls/model")
joblib.dump(time_xsmall_corr, "../saved_models/concentrations/xsmall_corr_pls/time")

