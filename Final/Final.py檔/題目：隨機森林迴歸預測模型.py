#!/usr/bin/env python
# coding: utf-8

# In[52]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV#自動調參
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt


# In[53]:


df = pd.read_csv('D:\IMLP342\data\\cars.csv')
df.head()


# In[54]:


plt.title('Cars', fontsize=15)
plt.xlabel("Speed",fontsize=10)
plt.ylabel("dist",fontsize=10)
plt.scatter(df['speed'], df['dist'])


# In[55]:


train, test = train_test_split(
    df, test_size = 0.3, random_state = 1)


# In[56]:


len(train)


# In[57]:


len(test)


# In[58]:


pd.isnull(df).sum()


# In[59]:


scaler = StandardScaler()

speed_train = np.array(train['speed']).reshape(-1,1)
speed_test = np.array(test['speed']).reshape(-1,1)
dist_train = np.array(train['dist']).reshape(-1,1)
dist_test = np.array(test['dist']).reshape(-1,1)

train['speed'] = scaler.fit_transform(speed_train)
test['speed'] = scaler.transform(speed_test)
train['dist'] = scaler.fit_transform(dist_train)
test['dist'] = scaler.transform(dist_test)

train.head()


# In[60]:


trainX = np.array(train['speed']).reshape(-1,1)
testX = np.array(test['speed']).reshape(-1,1)
trainY = np.array(train['dist']).reshape(-1,1)
testY = np.array(test['dist']).reshape(-1,1)


# In[61]:


# 建立 random forest 模型
forest = RandomForestRegressor(n_estimators = 100)
forest_fit = forest.fit(trainX, trainY)


# In[62]:


trainY_predicted = forest.predict(trainX)
testY_predicted = forest.predict(testX)
testY_predicted


# In[65]:


plt.scatter(trainX, trainY_predicted, c = 'red')
plt.scatter(trainX, trainY, c = 'gray', alpha = 0.5)


# In[32]:


plt.plot(trainX, trainY_predicted, color = 'red', linewidth = 3)
plt.plot(trainX, trainY, color = 'gray', linewidth = 3, alpha = 0.5)


# In[68]:


trainPredict = scaler.inverse_transform(trainY_predicted.reshape(-1,1))
trainY_val = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testY_predicted.reshape(-1,1))
testY_val = scaler.inverse_transform(testY)


# In[77]:


MSE_train = metrics.mean_squared_error(trainY_val, trainPredict)
print("Training Set--MSE = ", MSE_train)


# In[78]:


MSE_test = metrics.mean_squared_error(testY_val, testPredict)
print("Testing Set--MSE = ", MSE_test)


# In[80]:


plt.plot(testX, testY_predicted, color = 'red', linewidth = 3)
plt.plot(testX, testY, color = 'gray', linewidth = 3, alpha = 0.5)


# In[ ]:




