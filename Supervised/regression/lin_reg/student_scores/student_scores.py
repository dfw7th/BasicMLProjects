#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


# In[2]:


scores = pd.read_csv("/home/fw7th/Documents/ML_datasets/Linear_Regression/student_scores.csv")


# In[3]:


scores["TotalScore"] = scores["MathScore"] + scores["ReadingScore"] + scores["WritingScore"]


# In[4]:


data = scores.drop(["ReadingScore", "MathScore", "WritingScore"], axis=1)


# In[5]:


data.columns


# In[6]:


data.info()


# In[7]:


# Making the train and test set
train_set, test_set = train_test_split(data, test_size=0.3, random_state=42)


# In[8]:


train_set.head()


# In[9]:


train_set.dtypes[train_set.dtypes != 'object']


# In[10]:


plt.scatter(x="NrSiblings", y="TotalScore", data=train_set, alpha=0.4)


# In[11]:


stats.zscore(train_set['NrSiblings']).sort_values().tail(10)


# In[12]:


train_set.query("NrSiblings == 7 & TotalScore < 50")


# In[13]:


# Dropping our "Outliers"
train_set = train_set.drop([9268, 55, 18432], axis=0)


# In[14]:


# Getting the features with the most missing values
train_set.isnull().sum().sort_values(ascending=False).head(20)


# In[15]:


train_set["TransportMeans"].unique()


# In[16]:


train_set["ParentEduc"].unique()


# In[17]:


train_set["TransportMeans"].fillna("foot", inplace=True)
test_set["TransportMeans"].fillna("foot", inplace=True)


# In[18]:


train_set["ParentEduc"].fillna("No", inplace=True)
test_set["ParentEduc"].fillna("No", inplace=True)


# In[19]:


plt.figure(figsize=(20, 12))
sns.catplot(data=train_set, x="ParentEduc", y="TotalScore", kind="box", height=5, aspect=10/5)
plt.show()


# In[20]:


train_set["EthnicGroup"].unique()


# In[21]:


train_set["EthnicGroup"].fillna("No", inplace=True)
test_set["EthnicGroup"].fillna("No", inplace=True)


# In[22]:


sns.catplot(data=train_set, x="EthnicGroup", y="TotalScore", kind="box")


# In[23]:


train_set.isnull().sum().sort_values(ascending=False).head(20)


# In[24]:


test_set["TestPrep"].unique()


# In[25]:


train_set["TestPrep"].fillna("some", inplace=True)
test_set["TestPrep"].fillna("some", inplace=True)


# In[26]:


sns.catplot(data=train_set, x="TestPrep", y="TotalScore", kind="box")


# In[27]:


train_set["NrSiblings"].unique()


# In[28]:


sns.catplot(data=train_set, x="NrSiblings", y="TotalScore", kind="box")


# In[29]:


train_set["ParentMaritalStatus"].unique()


# In[30]:


train_set["ParentMaritalStatus"].fillna("dating", inplace=True)
test_set["ParentMaritalStatus"].fillna("dating", inplace=True)


# In[31]:


sns.catplot(data=train_set, x="ParentMaritalStatus", y="TotalScore", kind="box")


# In[32]:


train_set["WklyStudyHours"].unique()


# In[33]:


train_set["WklyStudyHours"].fillna("No", inplace=True)
test_set["WklyStudyHours"].fillna("No", inplace=True)


# In[34]:


sns.catplot(data=train_set, x="WklyStudyHours", y="TotalScore", kind="box")


# In[35]:


train_set["IsFirstChild"].unique()


# In[36]:


sns.catplot(data=train_set, x="IsFirstChild", y="TotalScore", kind="box")


# In[37]:


train_set["PracticeSport"].unique()


# In[38]:


sns.catplot(data=train_set, x="PracticeSport", y="TotalScore", kind="box")


# F.Engineering

# In[39]:


train_set = train_set.drop("Unnamed: 0", axis=1)
test_set = test_set.drop("Unnamed: 0", axis=1)


# In[40]:


train_set.head()


# In[41]:


train_set.corr(numeric_only=True)


# In[42]:


# Now normally we would start making features with existing features but we have no numerical values in this dataset


# In[43]:


sns.histplot(
    train_set,
    x=train_set["TotalScore"]
)


# In[44]:


train_set.dtypes[train_set.dtypes == 'object']


# In[45]:


num_cols = train_set.select_dtypes(include=["int64", "float64"]).columns
num_cols = num_cols.drop("TotalScore")


# In[46]:


ord_col = ["ParentEduc", "LunchType", "TestPrep", "PracticeSport", "TransportMeans", "WklyStudyHours"]


# In[47]:


ohe_col = ["Gender", "EthnicGroup", "ParentMaritalStatus", "IsFirstChild"]


# In[48]:


num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median"))
])


# In[49]:


ord_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])  


# In[50]:


ohe_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown='error', sparse_output=False))
])


# In[51]:


col_trans = ColumnTransformer(transformers=[
    ("num_p", num_pipe, num_cols), 
    ("ord_p", ord_pipe, ord_col),
    ("ohe_p", ohe_pipe, ohe_col),
    ], 
    remainder="passthrough", 
    n_jobs=-1)


# In[52]:


pipeline = Pipeline([
    ("preprocessing", col_trans)
])


# In[53]:


X = train_set.drop("TotalScore", axis=1)
y = train_set["TotalScore"]


# In[54]:


X_done = pipeline.fit_transform(X)


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X_done, y, random_state=42, test_size=0.20)


# In[56]:


lin_model = LinearRegression(n_jobs=-1)


# In[57]:


lin_model.fit(X_train, y_train)


# In[58]:


y_pred = lin_model.predict(X_test)


# In[78]:


lin_model.score(X_test, y_test)


# In[59]:


mean_squared_error(y_test, y_pred)


# In[62]:


param_grid = {
    'fit_intercept': [True, False],
    'positive': [True, False]
}


# In[71]:


grid_search = GridSearchCV(lin_model, 
                           param_grid, 
                           cv=10,
                           scoring="neg_mean_squared_error", 
                           n_jobs=-1)


# In[72]:


grid_model = grid_search.fit(X_train, y_train)


# In[80]:


best_score = grid_model.best_score_


# In[88]:


np.sqrt(-(best_score))


# In[89]:


best_model = grid_search.best_estimator_


# In[90]:


test_score = best_model.score(X_test, y_test)


# In[91]:


test_score


# In[ ]:




