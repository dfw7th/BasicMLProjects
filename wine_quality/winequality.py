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


# This is future David, after evaluating the model, we saw that the model doesn't accuately pick up on 8's, 3's and 4's (they have the least occurences in the data)
# So we should probably drop observations where quality is 3, 4, and 8
red["quality"].value_counts()

# We can see that they have the least occurences


# In[7]:


# Let's just drop the 3's and 8's and see what changes in out model
red = red.drop(red[red['quality'] == 3].index)
red = red.drop(red[red['quality'] == 8].index)
red = red.drop(red[red['quality'] == 4].index)


# In[8]:


# Ok we are trying to predict the quality feature
red.info()


# In[9]:


# Let's split the train and test sets here
train = red[0:481]
test = red[481:]


# In[10]:


# Now let's make a copy of the train set to work on so we don't affect it later, the test set as well
train_copy = train.copy()
test_copy = test.copy()


# In[11]:


train_copy.head()


# In[12]:


# No null values in the dataset so we move


# ## EDA and Analysis

# #### quality

# In[13]:


# Let's try to understand and see the target feature a little
train_copy["quality"].value_counts()


# In[14]:


# I think I'll drop the one observation with a 3, let's see it tho
train_copy.query("quality==3")

# oh why the wine is rated so poorly is because of the high citric acid content, I just checked and it makes wines taste sharp, sour and bad


# In[15]:


# Let's use a frame of reference to check if it's an outlier of some sorts
train_copy.query("quality==5").head()

# Don't worry let's keep it, who knows there might be other 3's in the test set


# #### fixed acidity

# In[16]:


sns.histplot(train_copy["fixed acidity"])
# Looks somewhat normal with a lot of the data in the median


# In[17]:


sns.boxplot(data=train_copy, x="quality", y="fixed acidity")

# Seems to be some kind of trend here, wines with a lower fixed acidity median, generally have to have lower scores


# In[18]:


# Let's do oneway anova, although we already see a difference in the classes, let's see it with math
from scipy.stats import f_oneway
statistic, pvalue = f_oneway(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# oh yh this a good feature for sure


# In[19]:


# Let's see spearman on it, since it is kind of a ranked feature, let's see how it would perform
from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# ok there does seem to be some positive correlation although it's a kinda weak, I think it's an important feature, so we'll keep it


# In[20]:


# Kruskal-wallis test is like ANOVA, but does not assume normality
from scipy.stats import kruskal
statistic, pvalue = kruskal(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# oh yh this a good feature for sure


# #### volatile acidity

# In[21]:


sns.histplot(train_copy["volatile acidity"])


# In[22]:


sns.boxplot(data=train_copy, x="quality", y="volatile acidity")

# Seems to be a trend here as well, kinda inverse proportional compared to the top one


# In[23]:


# Performing oneway ANOVA on this as well
statistic, pvalue = f_oneway(train_copy["quality"], train_copy["volatile acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")


# In[24]:


from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["volatile acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# Makes sense for the correlation to be negative based on the boxplot


# In[25]:


from scipy.stats import kruskal
statistic, pvalue = kruskal(train_copy["quality"], train_copy["fixed acidity"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")


# #### citric acid

# In[26]:


sns.histplot(train_copy["citric acid"])

# Kinda weirdly distributed


# In[27]:


sns.boxplot(data=train_copy, x="quality", y="citric acid")

# There does seem to be a trend here as well


# In[28]:


# Let's use kruskal on this since citric acid isn't placed normally at all
statistic, pvalue = kruskal(train_copy["quality"], train_copy["citric acid"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# ok nice variance with the categories


# In[29]:


from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["citric acid"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# Had kind of a rank correlation I guess


# #### residual sugar

# In[30]:


# Let's see the distribution of the feature
sns.histplot(train_copy["residual sugar"])


# In[31]:


sns.boxplot(data=train_copy, x="quality", y="residual sugar")

# There's a trend here although it's not as prominent as the others


# In[32]:


from scipy.stats import spearmanr
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["residual sugar"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# Correlation is low icl, but we'll see how the feature affects the model with RFE and check feature importances


# In[33]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["residual sugar"])

print(f"F-statistic: {statistic}")
print(f"Pvalue: {pvalue}")

# There's deviation in the classes tho, good thing


# #### chlorides

# In[34]:


train_copy.chlorides.dtype


# In[35]:


sns.histplot(train_copy["chlorides"])

# Seems normal but contains heavy outliers, we'll use kruskal-wallis test instead on one way ANOVA


# In[36]:


sns.boxplot(data=train_copy, x="quality", y="chlorides")


# In[37]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["chlorides"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# Just like I thought even from the boxplot, this feature doesn't really affect the quality of the wine, we'll decide with RFE and such


# In[38]:


# Let's see kruskal tho, there doesn't seem to be much class variation across the feature here, I can't lie
statistic, pvalue = kruskal(train_copy["quality"], train_copy["chlorides"])

print(f"kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")


# #### free sulfur dioxide

# In[39]:


train_copy["free sulfur dioxide"].dtype


# In[40]:


sns.histplot(train_copy["free sulfur dioxide"])

# Looks skewed postively skewed and it has outliers to top it off


# In[41]:


sns.boxplot(data=train_copy, x="quality", y="free sulfur dioxide")

# oh yh but there is some trend here, higher quality wines have lower sulphur content generally


# In[42]:


# Let's check the correlation between the features
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["free sulfur dioxide"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# Yh you could tell it would be a negative correlation at least, let's see kruskal


# In[43]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["free sulfur dioxide"])

print(f"kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")

# Yh nice, there is deviation in the feature that's for sure


# #### total sulfur dioxide

# In[44]:


# Checking the distriubtion of the feature
sns.histplot(train_copy["total sulfur dioxide"])

# Oh not bad actually, its a little positively skewed tho, I think one way ANOVA will suffice


# In[45]:


sns.boxplot(data=train_copy, x="quality", y="total sulfur dioxide")

# It doesn't look like there's a pattern here tho, we'll see


# In[46]:


# Let's check the correlation between the features
statistic, pvalue = spearmanr(train_copy["quality"], train_copy["total sulfur dioxide"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# Not bad actually, better than most features


# In[47]:


statistic, pvalue = f_oneway(train_copy["quality"], train_copy["total sulfur dioxide"])

print(f"ANOVA statistic: {statistic}")
print(f"pvalue: {pvalue}")

# Looking nice and peng


# #### density

# In[48]:


train_copy["density"].dtype


# In[49]:


sns.histplot(train_copy["density"])

# Wow, I'm impressed, it's pretty normally distributed


# In[50]:


sns.boxplot(data=train_copy, x="quality", y="density")

# Postivish trend I guess


# In[51]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["density"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")
# pvalue is relatively low compared to the other features


# In[52]:


statistic, pvalue = f_oneway(train_copy["quality"], train_copy["density"])

print(f"ANOVA statistic: {statistic}")
print(f"pvalue: {pvalue}")

# Oh no, well that does mean it might be neglible tho == very good


# In[53]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["total sulfur dioxide"])

print(f"kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")

# We'll see soon enough


# #### pH

# In[54]:


train_copy["pH"].dtype


# In[55]:


sns.histplot(train_copy["pH"])

# Wowwww really beautiful, closest thing I've ever seen to a gaussian like distribution since I started doing EDA in ML


# In[56]:


sns.boxplot(data=train_copy, x="quality", y="pH")

# yup, there's is a trend here


# In[57]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["pH"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")
# Sad looking correlation smh, we'll see later tho


# In[58]:


# Let's do ANOVA, I suspect the pvalue will be 0
statistic, pvalue = f_oneway(train_copy["quality"], train_copy["pH"])

print(f"ANOVA statistic: {statistic}")
print(f"pvalue: {pvalue}")

# yup just what I thought haha


# #### sulphates

# In[59]:


sns.histplot(train_copy["sulphates"])

# postitively skewed with some outliers


# In[60]:


sns.boxplot(data=train_copy, x="quality", y="sulphates")

# Oh, yh that's a beautiful looking postive trend/correlation


# In[61]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["sulphates"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# Yh I expected at least a somewhat present positive correlation


# In[62]:


statistic, pvalue = f_oneway(train_copy["quality"], train_copy["sulphates"])

print(f"ANOVA statistic: {statistic}")
print(f"pvalue: {pvalue}")


# In[63]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["pH"])

print(f"Kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")


# #### alcohol

# In[64]:


train_copy["alcohol"].dtype


# In[65]:


# Let's check out the distribution
sns.histplot(train_copy["alcohol"])

# Positively skewed 


# In[66]:


sns.boxplot(data=train_copy, x="quality", y="alcohol")

# I predict a strong postive correlation


# In[67]:


statistic, pvalue = spearmanr(train_copy["quality"], train_copy["alcohol"])

print(f"Correlation: {statistic}")
print(f"pvalue: {pvalue}")

# 0.4 is the strongest we've seen in a feature so far tho lol


# In[68]:


statistic, pvalue = kruskal(train_copy["quality"], train_copy["alcohol"])

print(f"Kruskal statistic: {statistic}")
print(f"pvalue: {pvalue}")


# In[69]:


# Nice done with that (for now)


# ## Feature Engineering

# In[70]:


# Let's analyze the features, check for multicollinearity and see if we can make more features  
plt.figure(figsize = (15,8))
sns.heatmap(train_copy.corr(numeric_only = True), annot = True, cmap= 'YlGnBu')


# In[71]:


# Let's make some features and see if we can get anything meaningful
train_copy["total acidity"] = train_copy["fixed acidity"] + train_copy["volatile acidity"]
train_copy["bound sulfur dioxide"] = train_copy["total sulfur dioxide"] - train_copy["free sulfur dioxide"]
train_copy["salts"] = train_copy["sulphates"] + train_copy["chlorides"]


# In[72]:


train_copy["alcohol"].unique()


# In[73]:


train_copy.head()


# In[74]:


# replotting the heatmap after creating some new features
plt.figure(figsize = (15,8))
sns.heatmap(train_copy.corr(numeric_only = True), annot = True, cmap= 'YlGnBu')


# In[75]:


# Let's drop total sulfur dioxide, salts
train_copy = train_copy.drop(["total sulfur dioxide", "salts"], axis=1)


# In[76]:


# Do the same for the test set
test_copy["total acidity"] = test_copy["fixed acidity"] + test_copy["volatile acidity"]
test_copy["bound sulfur dioxide"] = test_copy["total sulfur dioxide"] - test_copy["free sulfur dioxide"]
test_copy = test_copy.drop(["total sulfur dioxide"], axis=1)


# ## Model Training and Selection

# In[77]:


from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[78]:


# Let's define the two scalers we plan to use on the dataset, we will see how each of them performs on the dataset
robust = RobustScaler()
standard = StandardScaler()


# In[79]:


# Removing the target variable from the dataframe
X_copy = train_copy.drop("quality", axis=1)
y_copy = train_copy["quality"]


# In[80]:


# different X dataframes scaled with different scalers
X_copy_robust = robust.fit_transform(X_copy)
X_copy_standard = standard.fit_transform(X_copy)


# In[81]:


# One test and train set for robust scaler and another for standard scaler
X_copy_train_rob, X_copy_test_rob, y_copy_train, y_copy_test = train_test_split(X_copy_robust, y_copy, test_size=0.2, random_state=42)
X_copy_train_stand, X_copy_test_stand, y_copy_train, y_copy_test = train_test_split(X_copy_standard, y_copy, test_size=0.2, random_state=42)


# In[82]:


model1 = LogisticRegression(max_iter=1000, n_jobs=-1)
model2 = LogisticRegression(max_iter=1000, n_jobs=-1)


# In[83]:


# Let's train a model with the robust scaled X
model1.fit(X_copy_train_rob, y_copy_train)


# In[84]:


# The accuracy ain't too crazily bad, especially since it's multinominal we'll see precision and recall
pred1 = model1.predict(X_copy_test_rob)
accuracy_score(y_copy_test, pred1)


# In[85]:


# Let's see how the standard scaled one does
model2.fit(X_copy_train_stand, y_copy_train)


# In[86]:


pred2 = model2.predict(X_copy_test_stand)
accuracy_score(y_copy_test, pred2)

# Ok so they're the same, I'll do some feature scaling tho


# In[87]:


# Let's actually guage some metrics and see what's going on here
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc


# In[88]:


# creating the confusion matrix to see the TN, TP, FN, & FP
confusion_matrix(y_copy_test, pred2)


# In[89]:


# Just the confusion matrix but arranged more systematically
pd.crosstab(y_copy_test, pred2, rownames=['Actual'], colnames=['Predicted'], margins = True)


# In[90]:


# Classification report to see the metrics of our model
print(classification_report(y_copy_test, pred2))


# In[91]:


# This gives us the class probabilities for each observation, just neat to know
y_pred_proba = model2.predict_proba(X_copy_test_rob)


# In[92]:


# Found out that you can only plot the ROC curve and compute the auc score for binary classification unless you use OneVsRestClassifier,
# for multiclass classification problems, silly me


# In[93]:


# We are going to use our scaling methods to different features and shii and see how close of an accuracy we can get
# Let's see what we can do to get the best model and features and then we begin scaling and making the last and final model

# Let's do feature Importances to see how important each feature is to out model's final outcome
coefficients = model1.coef_[0]
odds_ratios = np.exp(coefficients)


# Display feature importance using coefficients and odds ratios
feature_importance = pd.DataFrame({
    'Feature': X_copy.columns,
    'Coefficient': coefficients,
    'Odds Ratio': odds_ratios
})
print("\nFeature Importance (Coefficient and Odds Ratio):")
print(feature_importance.sort_values(by='Coefficient', ascending=False))



# When looking at Feature Importances, in the coefficent, negative values are kinda like negative correlation with the log odds and vice versa
# So look for coefficients that are greater/less than 0 by a large margin, closer to 0 == "less significant feature"

# When looking at Odds Ratio look at features that are most deviated from 1, negatively or positively, closer to 1 = "less significant feature"

# Also, it's important to look at the p-values of the feature from feature selection, cuz if the coefficient is good but the p value is high, then...
# .. the feature might still be trash.

# Also consider domain knowledge with the features to know which are the most impactful.


# In[94]:


# ok now let's do RFE:
from sklearn.feature_selection import RFE
rfe = RFE(model1, n_features_to_select=9)

rfe.fit(X_copy_train_stand, y_copy_train)


rfe_features = X_copy.columns[rfe.support_]
print("\nSelected Features by RFE:")
print(rfe_features)

# RFE would return the best x features for your model


# In[95]:


# Let's drop "density", "residual sugar" and "free sulfur dioxide"
X_dropped = X_copy.drop(["density", "residual sugar", "free sulfur dioxide"], axis=1)


# In[96]:


# Let's standardize the new X
new_X = standard.fit_transform(X_dropped)


# In[97]:


new_train_x, new_test_x, new_train_y, new_test_y = train_test_split(new_X, y_copy, random_state=42, test_size=0.2)


# In[98]:


new_model = LogisticRegression(n_jobs=-1, max_iter=10000)
new_model.fit(new_train_x, new_train_y)


# In[99]:


preds = new_model.predict(new_test_x)


# In[100]:


accuracy_score(new_test_y, preds)

# Our model accuaracy dropped lollll, let's see the classification report since it's a three class model, accuracy doesn't tell the full story


# In[101]:


# Let's compare the two
print(classification_report(new_test_y, preds))


# In[102]:


# Just comparing the model2 report to this one

'''
    precision    recall  f1-score   support
    
               5       0.66      0.86      0.75        50
               6       0.57      0.41      0.48        39
               7       0.75      0.38      0.50         8
    
        accuracy                           0.64        97
       macro avg       0.66      0.55      0.58        97
    weighted avg       0.63      0.64      0.62        97

'''

# Oh no this one's actually better wow ok, noted, don't drop any features


# In[103]:


# For our final magic trick let's do a grid search and see the best hyperparameters for our model2
from sklearn.model_selection import GridSearchCV
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [.50, .75]
}

grid_search = GridSearchCV(model2, param_grid, scoring='accuracy', n_jobs=-1)


# In[104]:


grid_search.fit(X_copy_train_stand, y_copy_train)


# In[105]:


# Let's get the best regularization strength and penalty
print("Best regularization strength:", grid_search.best_params_['C'])
print("Best penalty:", grid_search.best_params_['penalty'])

if grid_search.best_params_['penalty'] == 'elasticnet':
    print("Best alpha:", grid_search.best_params_['l1_ratio'])

# This is basically the default log_reg hyperparameters, so I wanna try scaling some features and stuff, and see what happens


# ## Try to make the best model

# In[106]:


# Importing Libraries
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[107]:


# Making a log scaler
log_transform = FunctionTransformer(np.log1p)


# In[108]:


col = ColumnTransformer(
    transformers = [
        ("log", log_transform, ["bound sulfur dioxide", "alcohol"]),
        ("robust", RobustScaler(), ["sulphates", "density", "chlorides", "residual sugar", "volatile acidity"])
    ],
    remainder="passthrough",
    n_jobs=-1
)       


# In[109]:


pip1 = Pipeline([
    ("col", col),
    ("stand", StandardScaler()),
    ("log_reg", LogisticRegression(max_iter=10000, n_jobs=-1))
])


# In[110]:


# Let's split the fresh test and train set for this model
bestx_train, bestx_test, y_trainb, y_testb = train_test_split(X_copy, y_copy, random_state=42, test_size=0.2)


# In[111]:


pip1.fit(bestx_train, y_trainb)


# In[112]:


bst_pred = pip1.predict(bestx_test)


# In[113]:


accuracy_score(y_testb, bst_pred)

# Ok not a bad accuracy score for multiclass logistic regression


# In[114]:


# But the classification report determines everything
print(classification_report(y_testb, bst_pred))


# In[ ]:


# Let's compare it to the best one

'''
    precision    recall  f1-score   support
    
               5       0.66      0.86      0.75        50
               6       0.57      0.41      0.48        39
               7       0.75      0.38      0.50         8
    
        accuracy                           0.64        97
       macro avg       0.66      0.55      0.58        97
    weighted avg       0.63      0.64      0.62        97

'''


# In[116]:


# Let's make a pairplot and like see the distributions for all the features and redo it 
sns.pairplot(data=X_copy)


# In[117]:


# I definetly see some multicollinearity here smh, so let's drop "total acidity"
best_X = X_copy.drop("total acidity", axis=1)


# In[118]:


# now let's run it back
bstX_train, bstx_test, y_trainbst, y_testbst = train_test_split(best_X, y_copy, random_state=42, test_size=0.2)


# In[119]:


# We have to redo this so we have a brand new model 
pip2 = Pipeline([
    ("col", col),
    ("stand", StandardScaler()),
    ("log_reg", LogisticRegression(max_iter=10000, n_jobs=-1))
])


# In[120]:


pip2.fit(bstX_train, y_trainbst)


# In[122]:


bestpred = pip2.predict(bstx_test)


# In[124]:


accuracy_score(y_testbst, bestpred)

# Oh no it's just still worse


# In[126]:


# Let's see the classs report
print(classification_report(y_testbst, bestpred))


# In[ ]:


# Let's compare it to the best one

'''
    precision    recall  f1-score   support
    
               5       0.66      0.86      0.75        50
               6       0.57      0.41      0.48        39
               7       0.75      0.38      0.50         8
    
        accuracy                           0.64        97
       macro avg       0.66      0.55      0.58        97
    weighted avg       0.63      0.64      0.62        97

'''


# In[127]:


# I'm thinking let's just make one final pipeline and just apply robust scaler and standardize the set, let's see how it goes
pip_try = Pipeline([
    ("robb", RobustScaler()),
    ("stand", StandardScaler()),
    ("model_try", LogisticRegression(max_iter=1000, n_jobs=-1))
])


# In[128]:


# Let's test it on that X without mulitcollinearity and then the inital X
pip_try.fit(bstX_train, y_trainbst)


# In[129]:


final_try = pip_try.predict(bstx_test)


# In[130]:


accuracy_score(y_testbst, final_try)

# I think this is just the best we can obtain with such a basic model like logistic regression


# In[131]:


# Now with the initial X and y
tryX_train, tryx_test, y_traintry, y_testtry = train_test_split(X_copy, y_copy, random_state=42, test_size=0.2)


# In[132]:


pip_try2 = Pipeline([
    ("robb", RobustScaler()),
    ("stand", StandardScaler()),
    ("model_try", LogisticRegression(max_iter=1000, n_jobs=-1))
])


# In[133]:


pip_try.fit(tryX_train, y_traintry)


# In[134]:


try_pred = pip_try.predict(tryx_test)


# In[135]:


accuracy_score(y_testtry, try_pred)

# Yup this is just the best score we can hope to obtain with this model


# In[136]:


# Tomorrow we pick up with the final pipelining and model


# ## Final Pipeline and Model

# In[ ]:




