#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# In[4]:


data = pd.read_csv("C:\\Users\\neide\\OneDrive\\Escritorio\\6to Semestre\\framingham.csv")


# In[5]:


data


# In[6]:


data[['male','age']].head()


# In[7]:


data[['male','age']].plot.scatter(x='male',y='age')


# In[8]:


w = 0.09
b = -3.6


# In[9]:


x = np.linspace(0,data['male'].max(),100)
y = 1/(1+np.exp(-(w*x+b)))


# In[10]:


x = np.linspace(0,data['male'].max(),100)
y = 1/(1+np.exp(-(w*x+b)))


# In[11]:


data.plot.scatter(x='male',y='age')
plt.plot(x, y, color='black')
plt.ylim(0,Data['age'].max()*1.1)
plt.scatter(x, y, color='#A9E2F3')
# plt.grid()
plt.xlabel('Male')
plt.ylabel('age')
plt.show()


# In[12]:


data.plot.scatter(x='male',y='age')
plt.plot(x, y, color='black')
plt.ylim(0,Data['age'].max()*1.1)
plt.scatter(x, y, color='#A9E2F3')
# plt.grid()
plt.xlabel('Male')
plt.ylabel('age')
plt.show()


# In[13]:


data[['education','age']].plot.scatter(x='education',y='age')


# In[14]:


x = np.linspace(0,data['education'].max(),100)
y = 1/(1+np.exp(-(w*x+b)))


# In[15]:


data.plot.scatter(x='education',y='age')
plt.plot(x, y, color='black')
plt.ylim(0,data['age'].max()*1.1)
plt.scatter(x, y, color='#A9E2F3')
# plt.grid()
plt.xlabel('education')
plt.ylabel('age')
plt.show()


# In[16]:


data.plot.scatter(x='education',y='age')
plt.plot(x, y, color='black')
plt.ylim(0,data['age'].max()*1.1)
plt.scatter(x, y, color='#A9E2F3')
# plt.grid()
plt.xlabel('education')
plt.ylabel('age')
plt.show()


# In[ ]:




