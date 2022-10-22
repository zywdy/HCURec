#!/usr/bin/env python
# coding: utf-8
import os
import pickle

import numpy as np


data_root_path = 'data'
output_data_path = 'data'


with open(os.path.join(data_root_path, 'train', 'behaviors.tsv')) as f:
    behaviors1 = f.readlines()
with open(os.path.join(data_root_path, 'dev', 'behaviors.tsv')) as f:
    behaviors2 = f.readlines()


train_behaviors = []
val_behaviors = []
test_behaviors = []
index = np.random.permutation(len(behaviors1))
index_t = np.random.permutation(len(behaviors2))
index_sample = index[0: 60]
index_sample_v = index[60: 80]
index_sample_t = index_t[0: 20]
num = int(0.95 * len(behaviors1))
for i in index_sample:
    train_behaviors.append(behaviors1[i])
for i in index_sample_v:
    val_behaviors.append(behaviors1[i])
for i in index_sample_t:
    test_behaviors.append(behaviors2[i])
# for i in range(num):
#     train_behaviors.append(behaviors1[i])
# for i in range(num, len(behaviors1)):
#     val_behaviors.append(behaviors1[i])
# test_behaviors = behaviors2

# In[8]:


with open(os.path.join(output_data_path, 'train', 'behaviors.tsv'), 'w') as f:
    for i in range(len(train_behaviors)):
        f.write(train_behaviors[i])
with open(os.path.join(output_data_path, 'dev', 'behaviors.tsv'), 'w') as f:
    for i in range(len(val_behaviors)):
        f.write(val_behaviors[i])
with open(os.path.join(output_data_path, 'test', 'behaviors.tsv'), 'w') as f:
    for i in range(len(test_behaviors)):
        f.write(test_behaviors[i])