#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


reviews = pd.read_csv("/home/fw7th/Documents/ML_datasets/knn/amazon_rec/reviews_70k.csv", delimiter = "\t")


# In[9]:


train, test = train_test_split(reviews, random_state=123, test_size=0.3)


# In[10]:


train.head()


# In[11]:


train.info()


# In[12]:


# This one is gonna be a challenge but we have to push
# I think it's safe to drop the ID column here
train = train.drop("Id", axis=1)


# In[ ]:




