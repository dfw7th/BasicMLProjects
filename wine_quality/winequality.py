#!/usr/bin/env python
# coding: utf-8

# ## Loading Data

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# Let's load in the data
red_ini = pd.read_csv("/home/fw7th/Documents/ML_datasets/Logistic_Regression/wine_quality/winequality-red.csv")


# In[ ]:


red_ini.head(3) # Nah this shi is cooked, we are going to have to arrange the columns rq


# In[ ]:


red = pd.read_csv("/home/fw7th/Documents/ML_datasets/Logistic_Regression/wine_quality/winequality-red.csv", sep=";", skipinitialspace=True) 
# skipinitialspace to skip space after the delimeter


# In[ ]:


red.head(3) # Nicely done


# In[ ]:


# Ok we are trying to predict the quality feature
red.info()


# In[ ]:


# Let's split the train and test sets here
train = red[0:481]
test = red[481:]


# In[ ]:


# Now let's make a copy of the train set to work on so we don't affect it later
train_copy = train.copy()


# In[ ]:


train_copy.head()


# In[ ]:


# No null values in the dataset so we move


# ## EDA and Analysis

# #### quality

# In[ ]:


# Let's try to understand and see the target feature a little
train_copy["quality"].value_counts()


# In[ ]:


# I think I'll drop the one observation with a 3, let's see it tho
train_copy.query("quality==3")

# oh why the wine is rated so poorly is because of the high citric acid content, I just checked and it makes wines taste sharp, sour and bad


# In[ ]:


# Let's use a frame of reference to check if it's an outlier of some sorts
train_copy.query("quality==5").head()

# Don't worry let's keep it, who knows there might be other 3's in the test set


# #### fixed acidity

# In[ ]:


sns.histplot(train_copy["fixed acidity"])
# Looks somewhat normal with a lot of the data in the median


# In[ ]:


sns.boxplot(data=train_copy, x="quality", y="fixed acidity")

# Seems to be some kind of trend here, wines with a lower fixed acidity median, generally have to have lower scores


# In[ ]:


# Let's do oneway anova, although we already see a difference in the classes, let's see it with math
from scipy.stats import f_oneway
statistic, pvalue = f_oneway(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# oh yh this a good feature for sure


# In[ ]:


# Let's see spearman on it, since it is kind of a ranked feature, let's see how it would perform
from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# ok there does seem to be some positive correlation although it's a kinda weak, I think it's an important feature, so we'll keep it


# In[ ]:


# Kruskal-wallis test is like ANOVA, but does not assume normality
from scipy.stats import kruskal
statistic, pvalue = kruskal(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# oh yh this a good feature for sure


# #### volatile acidity

# In[ ]:


sns.histplot(train_copy["volatile acidity"])


# In[ ]:


sns.boxplot(data=train_copy, x="quality", y="volatile acidity")

# Seems to be a trend here as well, kinda inverse proportional compared to the top one


# In[ ]:


# Performing oneway ANOVA on this as well
statistic, pvalue = f_oneway(train_copy["quality"], train_copy["volatile acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")


# In[ ]:


from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["volatile acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# Makes sense for the correlation to be negative based on the boxplot


# In[ ]:


from scipy.stats import kruskal
statistic, pvalue = kruskal(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")


# #### citric acid

# In[ ]:


sns.histplot(train_copy["citric acid"])

# Kinda weirdly distributed


# In[ ]:


sns.boxplot(data=train_copy, x="quality", y="citric acid")

# There does seem to be a trend here as well


# In[ ]:


# Let's use kruskal on this since citric acid isn't placed normally at all
statistic, pvalue = kruskal(train_copy["quality"], train_copy["citric acid"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# ok nice variance with the categories


# In[ ]:


from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["citric acid"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# Had kind of a rank correlation I guess


# #### residual sugar

# In[ ]:


# Let's see the distribution of the feature
sns.histplot(train_copy["residual sugar"])


# In[ ]:


sns.boxplot(data=train_copy, x="quality", y="residual sugar")

# There's a trend here although it's not as prominent as the others


# In[ ]:


from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["residual sugar"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# Correlation is low icl, but we'll see how the feature affects the model with RFE and check feature importances


# In[ ]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["residual sugar"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# There's deviation in the classes tho, good thing


# #### chlorides

# In[ ]:


train_copy.chlorides.dtype


# In[ ]:


sns.histplot(train_copy["chlorides"])

# Seems normal but contains heavy outliers, we'll use kruskal-wallis test instead on one way ANOVA


# In[ ]:


sns.boxplot(data=train_copy, x="quality", y="chlorides")


# In[ ]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["chlorides"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# Just like I thought even from the boxplot, this feature doesn't really affect the quality of the wine, we'll decide with RFE and such


# In[ ]:


# Let's see kruskal tho, there doesn't seem to be much class variation across the feature here, I can't lie
statistic, pvalue = kruskal(train_copy["quality"], train_copy["chlorides"])

print(f"kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")


# #### free sulfur dioxide

# In[ ]:


train_copy["free sulfur dioxide"].dtype


# In[ ]:


sns.histplot(train_copy["free sulfur dioxide"])

# Looks skewed postively skewed and it has outliers to top it off


# In[ ]:


sns.boxplot(data=train_copy, x="quality", y="free sulfur dioxide")

# oh yh but there is some trend here, higher quality wines have lower sulphur content generally


# In[ ]:


# Let's check the correlation between the features
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["free sulfur dioxide"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# Yh you could tell it would be a negative correlation at least, let's see kruskal


# In[ ]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["free sulfur dioxide"])

print(f"kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")

# Yh nice, there is deviation in the feature that's for sure


# #### total sulfur dioxide

# In[ ]:


# Checking the distriubtion of the feature
sns.histplot(train_copy["total sulfur dioxide"])

# Oh not bad actually, its a little positively skewed tho, I think one way ANOVA will suffice


# In[ ]:


sns.boxplot(data=train_copy, x="quality", y="total sulfur dioxide")

# It doesn't look like there's a pattern here tho, we'll see


# In[ ]:


# Let's check the correlation between the features
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["total sulfur dioxide"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# Not bad actually, better than most features


# In[ ]:


statistic, pvalue = f_oneway(train_copy["quality"], train_copy["total sulfur dioxide"])

print(f"ANOVA statistic: {statistic}")
print(f"pvalue: {pvalue}")

# Looking nice and peng


# #### density

# In[ ]:


train_copy["density"].dtype


# In[ ]:


sns.histplot(train_copy["density"])

# Wow, I'm impressed, it's pretty normally distributed


# In[ ]:


sns.boxplot(data=train_copy, x="quality", y="density")

# Postivish trend I guess


# In[ ]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["density"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")
# pvalue is relatively low compared to the other features


# In[ ]:


statistic, pvalue = f_oneway(train_copy["quality"], train_copy["density"])

print(f"ANOVA statistic: {statistic}")
print(f"pvalue: {pvalue}")

# Oh no, well that does mean it might be neglible tho == very good


# In[ ]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["total sulfur dioxide"])

print(f"kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")

# We'll see soon enough


# #### pH

# In[ ]:


train_copy["pH"].dtype


# In[ ]:


sns.histplot(train_copy["pH"])

# Wowwww really beautiful, closest thing I've ever seen to a gaussian like distribution since I started doing EDA in ML


# In[ ]:


sns.boxplot(data=train_copy, x="quality", y="pH")

# yup, there's is a trend here


# In[ ]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["pH"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")
# Sad looking correlation smh, we'll see later tho


# In[ ]:


# Let's do ANOVA, I suspect the pvalue will be 0
statistic, pvalue = f_oneway(train_copy["quality"], train_copy["pH"])

print(f"ANOVA statistic: {statistic}")
print(f"pvalue: {pvalue}")

# yup just what I thought haha


# #### sulphates

# In[ ]:


sns.histplot(train_copy["sulphates"])

# postitively skewed with some outliers


# In[ ]:


sns.boxplot(data=train_copy, x="quality", y="sulphates")

# Oh, yh that's a beautiful looking postive trend/correlation


# In[ ]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["sulphates"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# Yh I expected at least a somewhat present positive correlation


# In[ ]:


statistic, pvalue = f_oneway(train_copy["quality"], train_copy["sulphates"])

print(f"ANOVA statistic: {statistic}")
print(f"pvalue: {pvalue}")


# In[ ]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["pH"])

print(f"Kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")


# #### alcohol

# In[ ]:


train_copy["alcohol"].dtype


# In[ ]:


# Let's check out the distribution
sns.histplot(train_copy["alcohol"])

# Positively skewed 


# In[ ]:


sns.boxplot(data=train_copy, x="quality", y="alcohol")

# I predict a strong postive correlation


# In[ ]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["alcohol"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# 0.4 is the strongest we've seen in a feature so far tho lol


# In[ ]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["alcohol"])

print(f"Kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")


# In[ ]:


# Nice done with that (for now)


# ## Feature Engineering

# In[ ]:


# Let's analyze the features, check for multicollinearity and see if we can make more features  
plt.figure(figsize = (15,8))
sns.heatmap(train_copy.corr(numeric_only = True), annot = True, cmap= 'YlGnBu')


# In[ ]:


# Let's make some features and see if we can get anything meaningful
train_copy["total acidity"] = train_copy["fixed acidity"] + train_copy["volatile acidity"]
train_copy["bound sulfur dioxide"] = train_copy["total sulfur dioxide"] - train_copy["free sulfur dioxide"]
train_copy["salts"] = train_copy["sulphates"] + train_copy["chlorides"]


# In[ ]:


train_copy["alcohol"].unique()


# In[ ]:


train_copy.head()


# In[ ]:


# replotting the heatmap after creating some new features
plt.figure(figsize = (15,8))
sns.heatmap(train_copy.corr(numeric_only = True), annot = True, cmap= 'YlGnBu')


# In[ ]:


# Let's drop total sulfur dioxide, salts
train_copy = train_copy.drop(["total sulfur dioxide", "salts"], axis=1)


# ## Model Training and Selection

# In[ ]:




