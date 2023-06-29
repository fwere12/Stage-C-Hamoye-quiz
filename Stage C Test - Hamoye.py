#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


#Loading in our data
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv")
df.head()


# In[4]:


#Describing our data
df.describe().T


# In[5]:


#dropping the 'stab' column as it will be represented by the 'stabf' column
df.drop('stab', axis = 1, inplace = True)


# In[6]:


#Preprocessing and vector/matrix slicing
X, y = df.iloc[:, :-1], df.iloc[:, -1]    


# In[7]:


#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[8]:


#Standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
normalized_train_df = scaler.fit_transform(X_train)
normalized_train_df = pd.DataFrame(normalized_train_df, columns = X_train.columns)

normalized_test_df = scaler.transform(X_test)
normalized_test_df = pd.DataFrame(normalized_test_df, columns = X_test.columns)


# In[9]:


#model selection
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state = 1)
forest.fit(normalized_train_df, y_train)
forest_pred = forest.predict(normalized_test_df)


# In[10]:


from sklearn.metrics import classification_report
print(classification_report(y_test, forest_pred, digits = 4))


# In[11]:


#Extra tree classifier
from sklearn.ensemble import ExtraTreesClassifier
tree = ExtraTreesClassifier(random_state=1)
tree.fit(normalized_train_df, y_train)
tree_pred = tree.predict(normalized_test_df)


# In[12]:


#using the classifiation report for extra tree
print(classification_report(y_test, tree_pred, zero_division = True, digits = 6))


# In[14]:


pip install xgboost


# In[17]:


#XGBoost

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Encode the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Initialize and fit the XGBoost classifier
extreme = XGBClassifier(random_state=1)
extreme.fit(normalized_train_df, y_train_encoded)

# Predict using the trained classifier
extreme_pred_encoded = extreme.predict(normalized_test_df)

# Decode the predicted labels
extreme_pred = label_encoder.inverse_transform(extreme_pred_encoded)


# In[18]:


#Classification
print(classification_report(y_test, extreme_pred, digits=4))


# In[20]:


pip install lightgbm


# In[21]:


#lightgbm
from lightgbm import LGBMClassifier
light = LGBMClassifier(random_state=1)
light.fit(normalized_train_df, y_train)
light_pred = light.predict(normalized_test_df)


# In[22]:


#Classification
print(classification_report(y_test, light_pred, digits = 4))


# In[23]:


#Hyperparameters
n_estimators = [50, 100, 300, 500, 1000]
min_samples_split = [2, 3, 5, 7, 9]
min_samples_leaf = [1, 2, 4, 6, 8]
max_features = ['auto', 'sqrt', 'log2', None] 
hyperparameter_grid = {'n_estimators': n_estimators,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}


# In[24]:


#Randomised Search Cross Validation
from sklearn.model_selection import RandomizedSearchCV

tree2 = ExtraTreesClassifier(random_state=1)
clf = RandomizedSearchCV(tree2, hyperparameter_grid, cv=5, n_iter=10, scoring = 'accuracy', n_jobs = -1, verbose = 1, random_state=1)
search_result = clf.fit(normalized_train_df, y_train)
     


# In[25]:


#Checking for the best parameter for the model
search_result.best_params_


# In[26]:


#experimenting with this parameter to test the model's performance
tuned_tree = ExtraTreesClassifier(n_estimators=1000, min_samples_split=2, 
                                 min_samples_leaf=8, max_features=None, random_state=1)
tuned_tree.fit(normalized_train_df, y_train)
tuned_tree_pred = tuned_tree.predict(normalized_test_df)


# In[27]:


#classification report for this hyperparameter tuning
print(classification_report(y_test, tuned_tree_pred, digits=4))


# In[28]:


#plot graph of feature importance
feat_importance = pd.Series(tuned_tree.feature_importances_, index=X.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.show()

