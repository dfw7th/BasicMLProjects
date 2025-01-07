#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid", palette="pastel") # for pretty plots
from sklearn.model_selection import train_test_split


# In[2]:


reviews = pd.read_csv("/home/fw7th/Documents/ML_datasets/knn/amazon_rec/reviews_70k.csv", delimiter = "\t")


# In[3]:


train, test = train_test_split(reviews, random_state=123, test_size=0.3)


# In[4]:


# let's make the copy
train_cp = train.copy()


# In[5]:


train_cp.head()


# In[6]:


train_cp.info()


# In[7]:


# This one is gonna be a challenge but we have to push
# I think it's safe to drop the ID column here
train_cp = train_cp.drop("Id", axis=1)


# #### ProductId, ProfileName, UserId - BOW
# #### Summary - TF-IDF
# #### Review - Word2Vec

# In[8]:


# I've decided the kind of feat.eng. to use on the features, yayy


# In[12]:


# Ok for the time series data, after asking gippity, it was a kind or Unix timestamps, or epoch timestamps, it represents
# seconds that have passed since January 1, 1970 at 00:00:00 UTC, weird ahh dataformat, we need to convert them to what we know

train_cp["Time"] = pd.to_datetime(train_cp['Time'], unit='s')
train_cp
# Yay let's go


# In[ ]:


# We'll pick up tmao

