#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[2]:


business = pd.read_csv("/home/fw7th/Documents/ML_datasets/shopping_trends.csv")


# In[3]:


business.columns


# In[4]:


business.describe()


# In[5]:


business.head()


# In[6]:


# Checking for null values in the business data
business.isnull().sum().sort_values(ascending=False).head()


# In[7]:


business.info()


# In[8]:


business.dtypes[business.dtypes == "object"]


# In[9]:


business["Gender"].unique()


# In[10]:


business["Item Purchased"].unique()


# In[11]:


business["Category"].unique()


# In[12]:


business["Location"].unique()


# In[13]:


business["Size"].unique()


# In[14]:


business["Color"].unique()


# In[15]:


business["Season"].unique()


# In[16]:


business["Subscription Status"].unique()


# In[17]:


business["Payment Method"].unique()


# In[18]:


business["Shipping Type"].unique()


# In[19]:


business["Discount Applied"].unique()


# In[20]:


business["Promo Code Used"].unique()


# In[21]:


business["Preferred Payment Method"].unique()


# In[22]:


business["Frequency of Purchases"].unique()


# In[23]:


# Checking the ANOVA of categorical variables in the feature "Gender"
from scipy.stats import f_oneway

# Example: Group the target by categories of the feature
groups = [business[business['Gender'] == category]['Review Rating'] for category in business['Gender'].unique()]
f_stat, p_value = f_oneway(*groups)
print("F-statistic:", f_stat)
print("p-value:", p_value)

# Larger F-value: A larger F-value indicates a greater difference among the group means. 
# It suggests that the variations between the groups are significant.

# Smaller F-value: Conversely, a smaller F-value suggests that the group means are similar, 
# and there may not be significant differences among them.


# The F-statistic, accompanied by a p-value, becomes your compass in ANOVA. 
# If the p-value is below a predetermined level (usually 0.05), 
# you can conclude that there are differences among the groups you’re comparing

# If the p-value is greater than the predetermined level then that means there isn't a significant difference between the groups


# This is applicable only in some cases. When you have more than two groups, and the ANOVA indicates significant differences, 
# you may want to perform post-hoc tests. These tests, such as Tukey’s HSD or Bonferroni correction, 
# can help you identify which specific groups are different from each other. They provide a more detailed view of where the differences lie.


# In[24]:


business.head()


# In[30]:


sns.histplot(data=business, x="Purchase Amount (USD)", bins=120)


# In[32]:


business["Purchase Amount (USD)"].unique()


# In[ ]:





# ## Really?

# In[ ]:





# In[ ]:





# OBJECTIVES
# 1) FIND A WAY TO IMPROVE CLIENT RATINGS
# 2) FIND A WAY TO IMPROVE PURCHASE AMOUNT
# 3) INCREASE THE PROBABILITY THAT A CUSTOMER WILL RETURN
# 4) WHAT ARE THE MOST COMMONLY AND THE LEAST BOUGHT ITEMS IN THE STORE, WHY, HOW CAN WE IMPROVE SALES OF THESE ITEMS?
# 5) AT WHAT TIMES OF THE YEAR ARE WHAT TYPES OF CLOTHES PURCHASED THE MOST, HOW DO WE LEVERAGE THIS?
# 6) WHAT DO OUR CLIENT'S SALES DEPEND THE MOST ON?
# 7) ARE SALES RATE AND PURCHASE AMOUNT LOCATION BASED?
# 8) HOW DO WE INCREASE FREQUENCY OF PURCHASES?
# 9) WHAT KIND OF CLOTHES, IN WHAT SIZE, COLOR ARE PURCHASED THE MOST, AND IN WHAT SEASON
# 10) DISCOUNTING??
# 11) WHAT PAYMENT METHOD GIVES THE HIGHEST AVERAGE PURCHASE AMOUNT?
# 12) SHIPPING?, MUST AFFECT PURCHASE AMOUNT FOR SURE.
# 13) HOW DOES THE AGE PURCHASING?
# 14) ARE THE STANDARDIZED PRICES OF THINGS, OR IS THIS MORE OF YOU CAN BARGAIN AND NEGOTIATE?
# 15) ARE THERE SEASONS WHERE THE PRICES OF SOME PARTICULAR ITEMS INCREASE
# 16) 
