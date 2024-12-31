#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats


# In[2]:


boston = pd.read_csv("~/Documents/ML_datasets/Linear_Regression/bostonHousing.csv")
boston.head()


# In[3]:


boston2 = boston.copy()


# In[4]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# In[5]:


y = boston2["MEDV"]
X = boston2.drop("MEDV", axis="columns")


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# In[7]:


model = LinearRegression().fit(X_train, y_train)


# In[8]:


y_pred = model.predict(X)


# In[9]:


model.score(X_test, y_test)


# In[10]:


MSE = mean_squared_error(y, y_pred)
print("RMSE: ", np.sqrt(MSE))


# In[ ]:





# PREPROCESSING

# In[11]:


bost = boston.copy()


# In[12]:


bost.info()


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[14]:


plt.figure(figsize=(17, 10))
sns.heatmap(bost.corr(), annot=True, cmap="YlGnBu")


# In[15]:


bost.describe()


# In[16]:


bost.plot(kind="box", figsize=(20, 15))


# In[17]:


bost.hist(bins=100, figsize=(20, 15))


# In[18]:


bost.plot(kind="scatter", x="TAX", y="AGE")


# In[19]:


bost.plot(kind="scatter", x="TAX", y="ZN")


# In[20]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(bost, test_size=0.25, random_state=42)


# In[21]:


housing_labels = train_set["MEDV"]
housing = train_set.drop("MEDV", axis=1)


# In[22]:


housing["NW"] = (housing.ZN * housing.RM)/housing.TAX


# In[23]:


housing.plot(kind="scatter", x="TAX", y="NW")


# In[24]:


cor = bost.corr()
cor["MEDV"].sort_values()


# In[25]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer


# In[26]:


logt = FunctionTransformer(np.log)
stand = StandardScaler()
robust = RobustScaler()


# In[27]:


age_to_robust = ["AGE"]
crim_to_log = ["CRIM"]


# In[28]:


scaling = ColumnTransformer([
    ("logg", logt, crim_to_log),
    ("rob", robust, age_to_robust),
])


# In[29]:


full_pipeline = Pipeline(
    steps=[
        ("scaling", scaling),
        ("stand_scale", stand)
    ]
)


# In[30]:


housing_prepared = full_pipeline.fit_transform(housing)


# In[31]:


from sklearn.linear_model import LinearRegression


# In[32]:


model = LinearRegression()
model.fit(housing_prepared, housing_labels)


# In[33]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", model.predict(some_data_prepared))


# In[34]:


print("Labels:", list(some_labels))


# In[35]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_prepared)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
print("RMSE: ", rmse)


# In[36]:


from sklearn.model_selection import cross_val_score
lin_scores = cross_val_score(
    model, housing_prepared, housing_labels, cv=3, scoring="neg_mean_squared_error"
)


# In[37]:


scores = (-lin_scores)
np.sqrt(scores)


# In[38]:


scores.mean()


# In[39]:


scores.std()


# In[40]:


X = test_set.drop("MEDV", axis=1)
y = test_set["MEDV"]


# In[41]:


X_ready = full_pipeline.fit_transform(X)


# In[43]:


model.score(X_ready, y)


# In[ ]:





# In[ ]:




