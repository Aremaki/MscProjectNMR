#!/usr/bin/env python
# coding: utf-8

# # Training of non-DL models for Multi-regression of the metabolites concentrations on indenpendent metabolites

# ### Install project packages

# In[6]:


get_ipython().run_cell_magic('bash', '', 'pip install -e ../.')


# ### Install required python modules

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'pip install -r ../requirements.txt')


# ### Import fucntions

# In[12]:


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

# ## I.1 Small independent dataset (1000 spectra)

# In[3]:


small_train_file_paths = ['../data/tfrecords/Concentrations_data/Small_sample/train/data_{}.tfrecord'.format(i)
                    for i in range(8)]
small_val_file_paths = ['../data/tfrecords/Concentrations_data/Small_sample/validation/data_{}.tfrecord'.format(i)
                    for i in range(2)]

small_train_dataset = read_tfrecords_concentrations(small_train_file_paths, 32)

small_val_dataset = read_tfrecords_concentrations(small_val_file_paths, 32)


# In[4]:


X_train_small = []
y_train_small = []

for element in small_train_dataset.unbatch():
    X_train_small.append(element[0].numpy())
    y_train_small.append(element[1].numpy())
    
for element in small_val_dataset.unbatch():
    X_train_small.append(element[0].numpy())
    y_train_small.append(element[1].numpy())

X_train_small = pd.DataFrame(X_train_small)
y_train_small = pd.DataFrame(y_train_small)


# ## I.2 Large independent dataset (10000 spectra)

# In[5]:


large_train_file_paths = ['../data/tfrecords/Concentrations_data/Large_sample/train/data_{}.tfrecord'.format(i) 
                        for i in range(32)]
large_val_file_paths = ['../data/tfrecords/Concentrations_data/Large_sample/validation/data_{}.tfrecord'.format(i) 
                      for i in range(8)]

large_train_dataset = read_tfrecords_concentrations(large_train_file_paths, 64)

large_val_dataset = read_tfrecords_concentrations(large_val_file_paths, 64)


# In[6]:


X_train_large = []
y_train_large = []

for element in large_train_dataset.unbatch():
    X_train_large.append(element[0].numpy())
    y_train_large.append(element[1].numpy())
    
for element in large_val_dataset.unbatch():
    X_train_large.append(element[0].numpy())
    y_train_large.append(element[1].numpy())

X_train_large = pd.DataFrame(X_train_large)
y_train_large = pd.DataFrame(y_train_large)


# ## I.3 Extra small independent dataset (100 spectra)

# In[7]:


xsmall_train_file_paths = ['../data/tfrecords/Concentrations_data/Extra_small_sample/train/data_{}.tfrecord'
                           .format(i) for i in range(4)]
xsmall_val_file_paths = ['../data/tfrecords/Concentrations_data/Extra_small_sample/validation/data_{}.tfrecord'
                         .format(i) for i in range(1)]

xsmall_train_dataset = read_tfrecords_concentrations(xsmall_train_file_paths, 16)

xsmall_val_dataset = read_tfrecords_concentrations(xsmall_val_file_paths, 16)


# In[8]:


X_train_xsmall = []
y_train_xsmall = []

for element in xsmall_train_dataset.unbatch():
    X_train_xsmall.append(element[0].numpy())
    y_train_xsmall.append(element[1].numpy())
    
for element in xsmall_val_dataset.unbatch():
    X_train_xsmall.append(element[0].numpy())
    y_train_xsmall.append(element[1].numpy())

X_train_xsmall = pd.DataFrame(X_train_xsmall)
y_train_xsmall = pd.DataFrame(y_train_xsmall)


# ---
# # II. Train Models

# ## II.1 Small independent dataset

# ### II.1.a Define mutli-output Random Forest

# In[9]:


small_multi_RF = RandomForestRegressor(10)


# ### II.1.b Fit mutli-output Random Forest

# In[15]:


if not os.path.exists("../saved_models/concentrations/small_multi_RF"):
    os.makedirs("../saved_models/concentrations/small_multi_RF")
start = time.time()
small_multi_RF.fit(X_train_small, y_train_small)
time_small = time.time() - start
joblib.dump(small_multi_RF, "../saved_models/concentrations/small_multi_RF/model")
joblib.dump(time_small, "../saved_models/concentrations/small_multi_RF/time")


# ### II.1.c Define single-output Random Forest

# In[17]:


small_inde_RFs = []

for i in range(48):
    small_inde_RFs.append(RandomForestRegressor(10))


# ### II.1.d Fit single-output Random Forest

# In[ ]:


for i in range(48):
    start = time.time()
    small_inde_RFs[i].fit(X_train_small, y_train_small[i])
    time_small_inde = time.time() - start
    if not os.path.exists("../saved_models/concentrations/small_single_RF/metabolite_{}".format(i)):
        os.makedirs("../saved_models/concentrations/small_single_RF/metabolite_{}".format(i))
    joblib.dump(small_inde_RFs[i], "../saved_models/concentrations/small_single_RF/metabolite_{}/model".format(i))
    joblib.dump(time_small_inde, "../saved_models/concentrations/small_single_RF/metabolite_{}/time".format(i))
    print('\n', '\n', '####', '\n', '\n', 'Model number {} has been trained in {} seconds !'
          .format(i, time_small_inde), '\n', '\n', '####', '\n', '\n')


# ### II.1.e Define partial least squares

# In[20]:


small_pls = PLSRegression(200)


# ### II.1.f Fit partial least squares

# In[ ]:


if not os.path.exists("../saved_models/concentrations/small_pls"):
    os.makedirs("../saved_models/concentrations/small_pls")
start = time.time()
small_pls.fit(X_train_small, y_train_small)
time_small = time.time() - start
joblib.dump(small_pls, "../saved_models/concentrations/small_pls/model")
joblib.dump(time_small, "../saved_models/concentrations/small_pls/time")


# ## II.2 Large independent dataset

# ### II.2.a Define mutli-output Random Forest

# In[29]:


large_multi_RF = RandomForestRegressor(10)


# ### II.2.b Fit mutli-output Random Forest

# In[ ]:


if not os.path.exists("../saved_models/concentrations/large_multi_RF"):
    os.makedirs("../saved_models/concentrations/large_multi_RF")
start = time.time()
large_multi_RF.fit(X_train_large, y_train_large)
time_large = time.time() - start
joblib.dump(large_multi_RF, "../saved_models/concentrations/large_multi_RF/model")
joblib.dump(time_large, "../saved_models/concentrations/large_multi_RF/time")


# ### II.2.c Define single-output Random Forest

# In[17]:


large_inde_RFs = []

for i in range(48):
    large_inde_RFs.append(RandomForestRegressor(10))


# ### II.2.d Fit single-output Random Forest

# In[ ]:


for i in range(48):
    start = time.time()
    large_inde_RFs[i].fit(X_train_large, y_train_large[i])
    time_large_inde = time.time() - start
    if not os.path.exists("../saved_models/concentrations/large_single_RF/metabolite_{}".format(i)):
        os.makedirs("../saved_models/concentrations/large_single_RF/metabolite_{}".format(i))
    joblib.dump(large_inde_RFs[i], "../saved_models/concentrations/large_single_RF/metabolite_{}/model".format(i))
    joblib.dump(time_large_inde, "../saved_models/concentrations/large_single_RF/metabolite_{}/time".format(i))
    print('\n', '\n', '####', '\n', '\n', 'Model number {} has been trained in {} seconds !'
          .format(i, time_large_inde), '\n', '\n', '####', '\n', '\n')


# ### II.2.e Define partial least squares

# In[20]:


large_pls = PLSRegression(200)


# ### II.2.f Fit partial least squares

# In[ ]:


if not os.path.exists("../saved_models/concentrations/large_pls"):
    os.makedirs("../saved_models/concentrations/large_pls")
start = time.time()
large_pls.fit(X_train_large, y_train_large)
time_large = time.time() - start
joblib.dump(large_pls, "../saved_models/concentrations/large_pls/model")
joblib.dump(time_large, "../saved_models/concentrations/large_pls/time")


# ## II.3 Exrtra small independent dataset

# ### II.3.a Define mutli-output Random Forest

# In[40]:


xsmall_multi_RF = RandomForestRegressor(10)


# ### II.3.b Fit mutli-output Random Forest

# In[ ]:


if not os.path.exists("../saved_models/concentrations/xsmall_multi_RF"):
    os.makedirs("../saved_models/concentrations/xsmall_multi_RF")
start = time.time()
xsmall_multi_RF.fit(X_train_xsmall, y_train_xsmall)
time_xsmall = time.time() - start
joblib.dump(xsmall_multi_RF, "../saved_models/concentrations/xsmall_multi_RF/model")
joblib.dump(time_xsmall, "../saved_models/concentrations/xsmall_multi_RF/time")


# ### II.3.c Define single-output Random Forest

# In[17]:


xsmall_inde_RFs = []

for i in range(48):
    xsmall_inde_RFs.append(RandomForestRegressor(10))


# ### II.3.d Fit single-output Random Forest

# In[ ]:


for i in range(48):
    start = time.time()
    xsmall_inde_RFs[i].fit(X_train_xsmall, y_train_xsmall[i])
    time_xsmall_inde = time.time() - start
    if not os.path.exists("../saved_models/concentrations/xsmall_single_RF/metabolite_{}".format(i)):
        os.makedirs("../saved_models/concentrations/xsmall_single_RF/metabolite_{}".format(i))
    joblib.dump(xsmall_inde_RFs[i], "../saved_models/concentrations/xsmall_single_RF/metabolite_{}/model".format(i))
    joblib.dump(time_xsmall_inde, "../saved_models/concentrations/xsmall_single_RF/metabolite_{}/time".format(i))
    print('\n', '\n', '####', '\n', '\n', 'Model number {} has been trained in {} seconds !'
          .format(i, time_xsmall_inde), '\n', '\n', '####', '\n', '\n')


# ### II.3.e Define partial least squares

# In[20]:


xsmall_pls = PLSRegression(200)


# ### II.3.f Fit partial least squares

# In[ ]:


if not os.path.exists("../saved_models/concentrations/xsmall_pls"):
    os.makedirs("../saved_models/concentrations/xsmall_pls")
start = time.time()
xsmall_pls.fit(X_train_xsmall, y_train_xsmall)
time_xsmall = time.time() - start
joblib.dump(xsmall_pls, "../saved_models/concentrations/xsmall_pls/model")
joblib.dump(time_xsmall, "../saved_models/concentrations/xsmall_pls/time")

