#!/usr/bin/env python
# coding: utf-8

# ## Loading Data

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Let's load in the data
red_ini = pd.read_csv("/home/fw7th/Documents/ML_datasets/Logistic_Regression/wine_quality/winequality-red.csv")


# In[3]:


red_ini.head(3) # Nah this shi is cooked, we are going to have to arrange the columns rq


# In[4]:


red = pd.read_csv("/home/fw7th/Documents/ML_datasets/Logistic_Regression/wine_quality/winequality-red.csv", sep=";", skipinitialspace=True) 
# skipinitialspace to skip space after the delimeter


# In[5]:


red.head(3) # Nicely done


# In[6]:


# Ok we are trying to predict the quality feature
red.info()


# In[7]:


# Let's split the train and test sets here
train = red[0:481]
test = red[481:]


# In[8]:


# Now let's make a copy of the train set to work on so we don't affect it later
train_copy = train.copy()


# In[10]:


train_copy.head()


# In[13]:


# No null values in the dataset so we move


# ## EDA and Analysis

# #### quality

# In[16]:


# Let's try to understand and see the target feature a little
train_copy["quality"].value_counts()


# In[24]:


# I think I'll drop the one observation with a 3, let's see it tho
train_copy.query("quality==3")

# oh why the wine is rated so poorly is because of the high citric acid content, I just checked and it makes wines taste sharp, sour and bad


# In[26]:


# Let's use a frame of reference to check if it's an outlier of some sorts
train_copy.query("quality==5").head()

# Don't worry let's keep it, who knows there might be other 3's in the test set


# #### fixed acidity

# In[42]:


sns.histplot(train_copy["fixed acidity"])
# Looks somewhat normal with a lot of the data in the median


# In[27]:


sns.boxplot(data=train_copy, x="quality", y="fixed acidity")

# Seems to be some kind of trend here, wines with a lower fixed acidity median, generally have to have lower scores


# In[30]:


# Let's do oneway anova, although we already see a difference in the classes, let's see it with math
from scipy.stats import f_oneway
statistic, pvalue = f_oneway(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# oh yh this a good feature for sure


# In[34]:


# Let's see spearman on it, since it is kind of a ranked feature, let's see how it would perform
from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# ok there does seem to be some positive correlation although it's a kinda weak, I think it's an important feature, so we'll keep it


# In[37]:


# Kruskal-wallis test is like ANOVA, but does not assume normality
from scipy.stats import kruskal
statistic, pvalue = kruskal(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# oh yh this a good feature for sure


# #### volatile acidity

# In[43]:


sns.histplot(train_copy["volatile acidity"])


# In[38]:


sns.boxplot(data=train_copy, x="quality", y="volatile acidity")

# Seems to be a trend here as well, kinda inverse proportional compared to the top one


# In[39]:


# Performing oneway ANOVA on this as well
statistic, pvalue = f_oneway(train_copy["quality"], train_copy["volatile acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")


# In[41]:


from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["volatile acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# Makes sense for the correlation to be negative based on the boxplot


# In[44]:


from scipy.stats import kruskal
statistic, pvalue = kruskal(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")


# #### citric acid

# In[45]:


sns.histplot(train_copy["citric acid"])

# Kinda weirdly distributed


# In[54]:


sns.boxplot(data=train_copy, x="quality", y="citric acid")

# There does seem to be a trend here as well


# In[63]:


# Let's use kruskal on this since citric acid isn't placed normally at all
statistic, pvalue = kruskal(train_copy["quality"], train_copy["citric acid"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# ok nice variance with the categories


# In[64]:


from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["citric acid"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# Had kind of a rank correlation I guess


# #### residual sugar

# In[65]:


sns.histplot(train_copy["residual sugar"])


# In[66]:


skan = np.log(train_copy["residual sugar"])
sns.histplot(skan)


# In[ ]:




