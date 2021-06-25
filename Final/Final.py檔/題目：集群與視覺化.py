#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('D:\IMLP342\data\iris.csv')
df.head()


# In[3]:


df['Species'].value_counts()


# In[4]:


pd.isnull(df).sum() #查看有無缺漏值


# In[5]:


encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species']) 

scaler = StandardScaler()

sepal_length = np.array(df['sepal.length']).reshape(-1,1)
sepal_width = np.array(df['sepal.width']).reshape(-1,1)
petal_length = np.array(df['petal.length']).reshape(-1,1)
petal_width = np.array(df['petal.width']).reshape(-1,1)

df["sepal.length"] = scaler.fit_transform(sepal_length)
df["sepal.width"] = scaler.fit_transform(sepal_width)
df["petal.length"] = scaler.fit_transform(petal_length)
df["petal.width"] = scaler.fit_transform(petal_width)


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(
     df.drop(['Species'],axis=1), df['Species'], test_size=0.3, random_state = 1)


# In[7]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(x_train, y_train)
train_predict = kmeans.predict(x_train)
test_predict = kmeans.predict(x_test)
print("test predict: ", test_predict)
print('\n')
print("test value: ", np.array(y_test))


# In[8]:


y_train = pd.DataFrame(y_train)
p_train = pd.DataFrame(train_predict, columns = ['Predict'])


# In[30]:


print(confusion_matrix(y_train,p_train))
print('\n')
print(classification_report(y_train,p_train))


# In[15]:


print(pd.crosstab(y_test, test_predict, 
           rownames = ["label"], colnames = ["predict"]))
print('\n')
print(classification_report(y_test, test_predict))


# In[11]:


dff = pd.read_csv('D:\IMLP342\data\iris.csv')
plt.scatter(dff['sepal.length'],dff['sepal.width'],c=df['Species'])


# In[12]:


plt.scatter(dff['sepal.length'],dff['sepal.width'],c=kmeans.predict(
     df.drop(['Species'],axis=1)))


# In[33]:


print(accuracy_score(y_train,p_train))


# In[34]:


print(accuracy_score(y_test, test_predict))

