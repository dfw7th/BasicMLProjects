#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


diabetes = pd.read_csv("/home/fw7th/Documents/ML_datasets/knn/diabetes.csv")


# In[3]:


diabetes.info()


# In[4]:


diabetes.tail()


# In[5]:


train = diabetes[0:530]
test = diabetes[530:]


# #### Pregnancies

# In[6]:


train["Pregnancies"].unique()


# In[7]:


from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train["Pregnancies"], train["Outcome"])
print(f"Corr = {statistic}")


# In[8]:


from scipy.stats import chi2_contingency
freq = pd.crosstab(train["Pregnancies"], train["Outcome"])
statistic, pvalue, dof, expected_freq = chi2_contingency(freq)
print(f"chi-statistic: {statistic}\n" 
      f"pvalue: {pvalue}\n" 
      f"dof: {dof}"
     )

# There is variance


# In[9]:


# There is variation tho
sns.barplot(data=train, x=train["Pregnancies"], y=train["Outcome"])


# In[10]:


train["Pregnancies"].value_counts()


# #### Glucose

# In[12]:


sns.histplot(train["Glucose"], bins=50)


# In[13]:


sns.boxplot(data=train, x="Outcome", y="Glucose")


# In[16]:


from scipy.stats import pointbiserialr
pointbiserialr(train["Outcome"], train["Glucose"])

# There is correlation btw the classes, although not extremely strong


# In[20]:


from scipy.stats import f_oneway
f_oneway(train["Outcome"], train["Glucose"])
         
# Very weird, let's try using the kruskal wallis test tho


# In[22]:


from scipy.stats import kruskal
kruskal(train["Outcome"], train["Glucose"])

# yup there's definitely difference of the classes


# #### BloodPressure

# In[24]:


sns.histplot(train["BloodPressure"], bins=50)

# Nice distribution fr


# In[25]:


sns.boxplot(data=train, x="Outcome", y="BloodPressure")

# The significance isn't glaring but it is there tho


# In[28]:


pointbiserialr(train["Outcome"], train["BloodPressure"])

# Correlation is nastily useless, we'll see tho, cuz we might drop this class


# In[32]:


kruskal(train["Outcome"], train["BloodPressure"])

# I don't trust this feature we'll still use RFE and feature selection methods tho


# #### SkinThickness

# In[33]:


sns.histplot(train["SkinThickness"], bins=50)


# In[ ]:




