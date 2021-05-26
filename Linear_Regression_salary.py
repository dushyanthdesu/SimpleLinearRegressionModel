#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas


# In[2]:


df=pandas.read_csv('SalaryData.csv')


# In[3]:


x=df["YearsExperience"].values.reshape(30,1)


# In[5]:


y=df["Salary"]


# In[6]:


from sklearn.linear_model import LinearRegression


# In[7]:


model=LinearRegression()


# In[8]:


model.fit(x,y)


# In[9]:


#coefficient
model.coef_


# In[10]:


#bias
model.intercept_


# In[11]:


model.predict([[2.6]])


# In[12]:


model.predict([[2.9]])


# In[13]:


#accuracy
53197.09/56642*100


# In[14]:


import joblib


# In[15]:


joblib.dump(model,'salary.pk1')


# In[ ]:




