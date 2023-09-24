#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn import linear_model # usando sklear para saber los valores optimos
import seaborn as sns
import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv("C:\\Users\\neide\\OneDrive\\Escritorio\\6to Semestre\\diabetes", sep=",")


# In[3]:


data = pd.read_csv("C:\\Users\\neide\\OneDrive\\Escritorio\\6to Semestre\\diabetes", sep=",")


# In[4]:


data = pd.read_csv("C:\\Users\\neide\\OneDrive\\Escritorio\\6to Semestre\\diabetes.csv", sep=",")


# In[5]:


data.info


# In[6]:


data.columns
data.info
data.describe


# In[7]:


data


# In[8]:


data.groupby(['Pregnancies']).count()['Glucose']


# In[9]:


data.plot.scatter(x="Age", y="Pregnancies")
plt.show()


# In[10]:


regresion = linear_model.LinearRegression()


# In[12]:


Ages = data["Age"].values.reshape((-1,1))


# In[13]:


modelo = regresion.fit(Age, data["Pregnancies"])


# In[14]:


print("Interseccion (b)", modelo.intercept_)
#imprimos la pendiente
print("Pendiente (m)", modelo.coef_)


# In[15]:


entrada= [[31],[50],[32],[21]]
predicciones = modelo.predict(entrada)
print(predicciones)


# In[16]:


data.plot.scatter(x="Age", y="Pregnancies", label='Datos originales')
plt.scatter(entrada, predicciones, color='red')
plt.plot(entrada, predicciones, color='black', label='Línea de regresión')
plt.xlabel('Age')
plt.ylabel('Pregnancies')
plt.legend()
plt.show()


# In[ ]:




