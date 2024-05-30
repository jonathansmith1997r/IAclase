#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
# In[2]:
wine = pd.read_csv('wine_data.csv', names = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", 
"Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", 
"Color_intensity", "Hue", "OD280", "Proline"])
# In[3]:
wine.head()
# In[4]:
wine.describe().transpose()
# In[5]:
# 178 data points with 13 features and 1 label column
wine.shape
# In[6]:
X = wine.drop('Cultivator',axis=1)
y = wine['Cultivator']
# In[7]:
from sklearn.model_selection import train_test_split
# In[8]:
X_train, X_test, y_train, y_test = train_test_split(X, y)
# In[9]:
from sklearn.preprocessing import StandardScaler
# In[10]:
scaler = StandardScaler()
# In[11]:
# Fit only to the training data
scaler.fit(X_train)
# In[12]:
StandardScaler(copy=True, with_mean=True, with_std=True)
# In[13]:
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# In[14]:
from sklearn.neural_network import MLPClassifier
# In[15]:
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
# In[19]:
from sklearn.metrics import classification_report,confusion_matrix
# In[20]:
print(confusion_matrix(y_test,predictions))
# In[21]:
print(classification_report(y_test,predictions))
