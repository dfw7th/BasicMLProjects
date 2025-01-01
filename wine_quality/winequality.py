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


# Now let's make a copy of the train set to work on so we don't affect it later, the test set as well
train_copy = train.copy()
test_copy = test.copy()


# In[9]:


train_copy.head()


# In[10]:


# No null values in the dataset so we move


# ## EDA and Analysis

# #### quality

# In[11]:


# Let's try to understand and see the target feature a little
train_copy["quality"].value_counts()


# In[12]:


# I think I'll drop the one observation with a 3, let's see it tho
train_copy.query("quality==3")

# oh why the wine is rated so poorly is because of the high citric acid content, I just checked and it makes wines taste sharp, sour and bad


# In[13]:


# Let's use a frame of reference to check if it's an outlier of some sorts
train_copy.query("quality==5").head()

# Don't worry let's keep it, who knows there might be other 3's in the test set


# #### fixed acidity

# In[14]:


sns.histplot(train_copy["fixed acidity"])
# Looks somewhat normal with a lot of the data in the median


# In[15]:


sns.boxplot(data=train_copy, x="quality", y="fixed acidity")

# Seems to be some kind of trend here, wines with a lower fixed acidity median, generally have to have lower scores


# In[16]:


# Let's do oneway anova, although we already see a difference in the classes, let's see it with math
from scipy.stats import f_oneway
statistic, pvalue = f_oneway(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# oh yh this a good feature for sure


# In[17]:


# Let's see spearman on it, since it is kind of a ranked feature, let's see how it would perform
from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# ok there does seem to be some positive correlation although it's a kinda weak, I think it's an important feature, so we'll keep it


# In[18]:


# Kruskal-wallis test is like ANOVA, but does not assume normality
from scipy.stats import kruskal
statistic, pvalue = kruskal(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# oh yh this a good feature for sure


# #### volatile acidity

# In[19]:


sns.histplot(train_copy["volatile acidity"])


# In[20]:


sns.boxplot(data=train_copy, x="quality", y="volatile acidity")

# Seems to be a trend here as well, kinda inverse proportional compared to the top one


# In[21]:


# Performing oneway ANOVA on this as well
statistic, pvalue = f_oneway(train_copy["quality"], train_copy["volatile acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")


# In[22]:


from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["volatile acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# Makes sense for the correlation to be negative based on the boxplot


# In[23]:


from scipy.stats import kruskal
statistic, pvalue = kruskal(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")


# #### citric acid

# In[24]:


sns.histplot(train_copy["citric acid"])

# Kinda weirdly distributed


# In[25]:


sns.boxplot(data=train_copy, x="quality", y="citric acid")

# There does seem to be a trend here as well


# In[26]:


# Let's use kruskal on this since citric acid isn't placed normally at all
statistic, pvalue = kruskal(train_copy["quality"], train_copy["citric acid"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# ok nice variance with the categories


# In[27]:


from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["citric acid"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# Had kind of a rank correlation I guess


# #### residual sugar

# In[28]:


# Let's see the distribution of the feature
sns.histplot(train_copy["residual sugar"])


# In[29]:


sns.boxplot(data=train_copy, x="quality", y="residual sugar")

# There's a trend here although it's not as prominent as the others


# In[30]:


from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["residual sugar"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# Correlation is low icl, but we'll see how the feature affects the model with RFE and check feature importances


# In[31]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["residual sugar"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# There's deviation in the classes tho, good thing


# #### chlorides

# In[32]:


train_copy.chlorides.dtype


# In[33]:


sns.histplot(train_copy["chlorides"])

# Seems normal but contains heavy outliers, we'll use kruskal-wallis test instead on one way ANOVA


# In[34]:


sns.boxplot(data=train_copy, x="quality", y="chlorides")


# In[35]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["chlorides"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# Just like I thought even from the boxplot, this feature doesn't really affect the quality of the wine, we'll decide with RFE and such


# In[36]:


# Let's see kruskal tho, there doesn't seem to be much class variation across the feature here, I can't lie
statistic, pvalue = kruskal(train_copy["quality"], train_copy["chlorides"])

print(f"kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")


# #### free sulfur dioxide

# In[37]:


train_copy["free sulfur dioxide"].dtype


# In[38]:


sns.histplot(train_copy["free sulfur dioxide"])

# Looks skewed postively skewed and it has outliers to top it off


# In[39]:


sns.boxplot(data=train_copy, x="quality", y="free sulfur dioxide")

# oh yh but there is some trend here, higher quality wines have lower sulphur content generally


# In[40]:


# Let's check the correlation between the features
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["free sulfur dioxide"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# Yh you could tell it would be a negative correlation at least, let's see kruskal


# In[41]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["free sulfur dioxide"])

print(f"kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")

# Yh nice, there is deviation in the feature that's for sure


# #### total sulfur dioxide

# In[42]:


# Checking the distriubtion of the feature
sns.histplot(train_copy["total sulfur dioxide"])

# Oh not bad actually, its a little positively skewed tho, I think one way ANOVA will suffice


# In[43]:


sns.boxplot(data=train_copy, x="quality", y="total sulfur dioxide")

# It doesn't look like there's a pattern here tho, we'll see


# In[44]:


# Let's check the correlation between the features
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["total sulfur dioxide"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# Not bad actually, better than most features


# In[45]:


statistic, pvalue = f_oneway(train_copy["quality"], train_copy["total sulfur dioxide"])

print(f"ANOVA statistic: {statistic}")
print(f"pvalue: {pvalue}")

# Looking nice and peng


# #### density

# In[46]:


train_copy["density"].dtype


# In[47]:


sns.histplot(train_copy["density"])

# Wow, I'm impressed, it's pretty normally distributed


# In[48]:


sns.boxplot(data=train_copy, x="quality", y="density")

# Postivish trend I guess


# In[49]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["density"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")
# pvalue is relatively low compared to the other features


# In[50]:


statistic, pvalue = f_oneway(train_copy["quality"], train_copy["density"])

print(f"ANOVA statistic: {statistic}")
print(f"pvalue: {pvalue}")

# Oh no, well that does mean it might be neglible tho == very good


# In[51]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["total sulfur dioxide"])

print(f"kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")

# We'll see soon enough


# #### pH

# In[52]:


train_copy["pH"].dtype


# In[53]:


sns.histplot(train_copy["pH"])

# Wowwww really beautiful, closest thing I've ever seen to a gaussian like distribution since I started doing EDA in ML


# In[54]:


sns.boxplot(data=train_copy, x="quality", y="pH")

# yup, there's is a trend here


# In[55]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["pH"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")
# Sad looking correlation smh, we'll see later tho


# In[56]:


# Let's do ANOVA, I suspect the pvalue will be 0
statistic, pvalue = f_oneway(train_copy["quality"], train_copy["pH"])

print(f"ANOVA statistic: {statistic}")
print(f"pvalue: {pvalue}")

# yup just what I thought haha


# #### sulphates

# In[57]:


sns.histplot(train_copy["sulphates"])

# postitively skewed with some outliers


# In[58]:


sns.boxplot(data=train_copy, x="quality", y="sulphates")

# Oh, yh that's a beautiful looking postive trend/correlation


# In[59]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["sulphates"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# Yh I expected at least a somewhat present positive correlation


# In[60]:


statistic, pvalue = f_oneway(train_copy["quality"], train_copy["sulphates"])

print(f"ANOVA statistic: {statistic}")
print(f"pvalue: {pvalue}")


# In[61]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["pH"])

print(f"Kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")


# #### alcohol

# In[62]:


train_copy["alcohol"].dtype


# In[63]:


# Let's check out the distribution
sns.histplot(train_copy["alcohol"])

# Positively skewed 


# In[64]:


sns.boxplot(data=train_copy, x="quality", y="alcohol")

# I predict a strong postive correlation


# In[65]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["alcohol"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# 0.4 is the strongest we've seen in a feature so far tho lol


# In[66]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["alcohol"])

print(f"Kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")


# In[67]:


# Nice done with that (for now)


# ## Feature Engineering

# In[68]:


# Let's analyze the features, check for multicollinearity and see if we can make more features  
plt.figure(figsize = (15,8))
sns.heatmap(train_copy.corr(numeric_only = True), annot = True, cmap= 'YlGnBu')


# In[69]:


# Let's make some features and see if we can get anything meaningful
train_copy["total acidity"] = train_copy["fixed acidity"] + train_copy["volatile acidity"]
train_copy["bound sulfur dioxide"] = train_copy["total sulfur dioxide"] - train_copy["free sulfur dioxide"]
train_copy["salts"] = train_copy["sulphates"] + train_copy["chlorides"]


# In[70]:


train_copy["alcohol"].unique()


# In[71]:


train_copy.head()


# In[72]:


# replotting the heatmap after creating some new features
plt.figure(figsize = (15,8))
sns.heatmap(train_copy.corr(numeric_only = True), annot = True, cmap= 'YlGnBu')


# In[73]:


# Let's drop total sulfur dioxide, salts
train_copy = train_copy.drop(["total sulfur dioxide", "salts"], axis=1)


# In[74]:


# Do the same for the test set
test_copy["total acidity"] = test_copy["fixed acidity"] + test_copy["volatile acidity"]
test_copy["bound sulfur dioxide"] = test_copy["total sulfur dioxide"] - test_copy["free sulfur dioxide"]
test_copy = test_copy.drop(["total sulfur dioxide"], axis=1)


# ## Model Training and Selection

# In[75]:


from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[76]:


# Let's define the two scalers we plan to use on the dataset, we will see how each of them performs on the dataset
robust = RobustScaler()
standard = StandardScaler()


# In[77]:


# Removing the target variable from the dataframe
X_copy = train_copy.drop("quality", axis=1)
y_copy = train_copy["quality"]


# In[78]:


# different X dataframes scaled with different scalers
X_copy_robust = robust.fit_transform(X_copy)
X_copy_standard = standard.fit_transform(X_copy)


# In[79]:


# One test and train set for robust scaler and another for standard scaler
X_copy_train_rob, X_copy_test_rob, y_copy_train, y_copy_test = train_test_split(X_copy_robust, y_copy, test_size=0.2, random_state=42)
X_copy_train_stand, X_copy_test_stand, y_copy_train, y_copy_test = train_test_split(X_copy_standard, y_copy, test_size=0.2, random_state=42)


# In[80]:


model1 = LogisticRegression(max_iter=1000, n_jobs=-1)
model2 = LogisticRegression(max_iter=1000, n_jobs=-1)


# In[81]:


# Let's train a model with the robust scaled X
model1.fit(X_copy_train_rob, y_copy_train)


# In[82]:


# The accuracy ain't too crazily bad, especially since it's multinominal we'll see precision and recall
pred1 = model1.predict(X_copy_test_rob)
accuracy_score(y_copy_test, pred1)


# In[83]:


# Let's see how the standard scaled one does
model2.fit(X_copy_train_stand, y_copy_train)


# In[85]:


pred2 = model2.predict(X_copy_test_stand)
accuracy_score(y_copy_test, pred2)
# Oh robust scaler works better by a minute quality I guess


# In[86]:


# Just said let me go on a little detour and see how log transform would be applied
from sklearn.preprocessing import FunctionTransformer


# In[87]:


log_model = FunctionTransformer(func=np.log1p)


# In[88]:


model3 = LogisticRegression(n_jobs=-1, max_iter=1000)


# In[90]:


X_copy_log = log_model.fit_transform(X_copy)


# In[91]:


X_copy_train_log, X_copy_test_log, y_copy_train, y_copy_test = train_test_split(X_copy_log, y_copy, test_size=0.2, random_state=42)


# In[92]:


model3.fit(X_copy_train_log, y_copy_train)


# In[95]:


log_pred = model3.predict(X_copy_test_log)
accuracy_score(y_copy_test, log_pred)
# I got decently surprised, ok so now I have and ides


# In[99]:


# Let's actually guage some metrics and see what's going on here
from sklearn.metrics import precision_recall_fscore_support


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# We are going to use our scaling methods to different features and shii and see how close of an accuracy we can get

