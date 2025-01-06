#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# In[2]:


medical_data = pd.read_csv("/home/fw7th/Documents/ML_datasets/Linear_Regression/medical_cost.csv")


# In[3]:


train_df, test_df = train_test_split(medical_data, test_size=0.2, random_state=42)


# In[4]:


train_df.info()


# In[5]:


train_df.dtypes[train_df.dtypes != "object"]


# In[6]:


plt.scatter(data=train_df, y="charges", x="age", alpha=0.3)


# In[7]:


train_df.query("charges > 50000")


# In[8]:


# Drop 1230, 819, 1300, 577, 34


# In[9]:


plt.scatter(data=train_df, y="charges", x="bmi", alpha=0.3)


# In[10]:


plt.scatter(data=train_df, y="charges", x="children", alpha=0.3)


# In[11]:


train_df.query("children == 4 & charges > 30000")


# In[12]:


#Drop 1012, 621


# In[13]:


train_df = train_df.drop([1012, 621, 1230, 819, 1300, 577, 34])


# In[14]:


# Getting the features with the most missing values
train_df.isnull().sum().sort_values(ascending=False).head(7)


# In[15]:


train_df.dtypes[train_df.dtypes == 'object']


# In[16]:


train_df["sex"].unique()


# In[17]:


sns.catplot(x="sex", y="charges", data=train_df, kind="box")


# In[18]:


train_df["smoker"].unique()


# In[19]:


sns.catplot(x="smoker", y="charges", data=train_df, kind="box")


# In[20]:


train_df["region"].unique()


# In[21]:


sns.catplot(x="region", y="charges", data=train_df, kind="box")


# F.Eng

# In[22]:


plt.figure(figsize=(12, 8))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap="YlGnBu")


# In[23]:


train_df.columns


# In[24]:


# Useful features
train_df["WperAge"] = train_df.bmi * train_df.age


# In[25]:


train_df.corr(numeric_only=True)


# In[26]:


# Useful features
train_df["WperAge"] = train_df.bmi * train_df.age
test_df["WperAge"] = test_df.bmi * test_df.age


# In[27]:


# Checking the ANOVA of categorical variables in the feature "sex"
from scipy.stats import f_oneway

# Example: Group the target by categories of the feature
groups = [train_df[train_df['sex'] == category]['charges'] for category in train_df['sex'].unique()]
f_stat, p_value = f_oneway(*groups)
print("F-statistic:", f_stat)
print("p-value:", p_value)


# In[28]:


# Checking "region"
groups = [train_df[train_df['region'] == category]['charges'] for category in train_df['region'].unique()]
f_stat, p_value = f_oneway(*groups)
print("F-statistic:", f_stat)
print("p-value:", p_value)


# In[29]:


# Checking "smoker"
groups = [train_df[train_df['smoker'] == category]['charges'] for category in train_df['smoker'].unique()]
f_stat, p_value = f_oneway(*groups)
print("F-statistic:", f_stat)
print("p-value:", p_value)


# In[30]:


train_df.columns


# In[31]:


# We are going to drop "region", "children", "sex"


# In[32]:


train_df["bmi"].hist(bins=50)


# In[33]:


train_df = train_df.drop(["region", "children", "sex"], axis=1)
test_df = test_df.drop(["region", "children", "sex"], axis=1)


# In[34]:


sns.histplot(
    train_df,
    x=train_df["charges"]
)


# In[35]:


train_df["charges"] = np.log1p(train_df["charges"])


# In[36]:


sns.histplot(
    train_df,
    x=train_df["charges"]
)


# In[37]:


#Making Pipelines and shii


# In[38]:


train_df.head()


# In[39]:


X = train_df.drop("charges", axis=1)
y = train_df["charges"]


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[41]:


num_col = ["WperAge", "bmi", "age"]


# In[42]:


ord_col = ["smoker"]


# In[43]:


num_pip = Pipeline([
    ("stand", StandardScaler()),
])


# In[44]:


ord_pip = Pipeline([
    ("ord", OrdinalEncoder())
])


# In[45]:


col_trans = ColumnTransformer(transformers=[
    ("num_col", num_pip, num_col),
    ("ord_col", ord_pip, ord_col),
], 
                              remainder="passthrough",
                              n_jobs = -1)


# In[46]:


full_pipeline = Pipeline([
    ("preprocessing", col_trans)
])


# In[47]:


X_done = full_pipeline.fit_transform(X_train)


# In[48]:


model = LinearRegression(n_jobs=-1)


# In[49]:


model.fit(X_done, y_train)


# In[50]:


X_test_done = full_pipeline.transform(X_test)


# In[51]:


y_pred = model.predict(X_test_done)


# In[52]:


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse


# In[56]:


model.score(X_test_done, y_test)


# In[53]:


X_valid = test_df.drop("charges", axis=1)
y_valid = np.log1p(test_df["charges"])


# In[54]:


X_valid_done = full_pipeline.transform(X_valid)


# In[55]:


model.score(X_valid_done, y_valid)


# In[68]:


y_valid_pred = model.predict(X_valid_done)


# In[69]:


mse_valid = mean_squared_error(y_valid, y_valid_pred)
rmse_valid = np.sqrt(mse_valid)
rmse_valid

