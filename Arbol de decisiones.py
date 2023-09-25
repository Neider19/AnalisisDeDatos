#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv("C:\\Users\\neide\OneDrive\\Escritorio\\6to Semestre\\framingham.csv")


# In[5]:


data.head()


# In[6]:


predictors_col=['male','age','education','BMI','glucose']
target_col=['diabetes']


# In[7]:


x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size=0.2,random_state=13)


# In[8]:


predictors = data [predictors_col]
target = data[target_col]


# In[9]:


x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size=0.2,random_state=13)


# In[10]:


tree = decisionTreeClassifer()


# In[11]:


from sklearn.tree import DecisionTreeClassifier


# In[12]:


tree = DecisionTreeClassifer()


# In[13]:


tree = decisiontreeClassifer()


# In[14]:


tree = DecisiontreeClassifer()


# In[15]:


tree = DecisiontreeClassifier()


# In[16]:


tree = DecisionTreeClassifier()


# In[17]:


arbol = tree.fit(x_train,y_train)


# In[18]:


plot_tree(arbol)


# In[19]:


from sklearn.tree import plot_tree
plot_tree(arbol)


# In[20]:


prediciones = arbol.predict(x_test)


# In[21]:


accuracy = accuracy_score(y_test,predicciones)


# In[22]:


predicciones = arbol.predict(x_test)


# In[23]:


accuracy = accuracy_score(y_test,predicciones)


# In[24]:


accuracy


# In[ ]:




