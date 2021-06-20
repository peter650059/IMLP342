#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers.recurrent import SimpleRNN
from datetime import datetime
import matplotlib.dates as mdates


# In[ ]:





# In[2]:


df = pd.read_csv('data\owl_world.csv')


# In[3]:


df.tail(10)


# In[4]:


date = df['Date'].values
cases = df['New_cases'].values
plt.plot(date, cases , color = 'blue')
plt.title('New_cases')
plt.show()


# In[5]:


col =['Date', 'New_cases']
df2 = df[col]
df2.tail(10)


# In[6]:


def create_dataset (dataset, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i +look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i +look_back, 0])
    return np.array(dataX), np.array(dataY)


# In[7]:


rate = 0.98
train_size = int(len(cases) * rate)
test_size = len(cases) - train_size
train, test = cases[0:train_size], cases[train_size:len(cases)]


# In[8]:


#建立MinMaxScaler物件
minmax = MinMaxScaler()
# 資料標準化
train_minmax = minmax.fit_transform(train.reshape(-1,1))
test_minmax = minmax.transform(test.reshape(-1,1))


# In[9]:


look_back = 1
trainX, trainY = create_dataset(train_minmax, look_back)
testX, testY = create_dataset(test_minmax, look_back)


# In[10]:


trainX = trainX.reshape(len(trainX), 1, trainX.shape[1])
testX = testX.reshape(len(testX), 1, testX.shape[1])


# # RNN
# ![image-2.png](attachment:image-2.png)
# 

# In[11]:


model = Sequential()


# In[12]:


model.add(SimpleRNN(50, input_shape = (trainX.shape[1:]), return_sequences=True))


# In[13]:


model.add(Dropout(0.2))


# In[14]:


model.add(SimpleRNN(50, return_sequences=True))


# In[15]:


model.add(Dropout(0.2))


# In[16]:


model.add(SimpleRNN(50, return_sequences=True))


# In[17]:


model.add(Dropout(0.2))


# In[18]:


model.add(SimpleRNN(50))


# In[19]:


model.add(Dropout(0.2))


# In[20]:


model.add(Dense(units=1,
                activation='relu' ))


# In[21]:


model.summary()


# In[22]:


model.compile(loss='mean_squared_error',
              optimizer='rmsprop')


# In[23]:


train_history = model.fit(trainX, trainY,
                          batch_size=10, 
                          epochs=50,
                          verbose=1)


# In[24]:


def show_train_history(loss):
    plt.plot(train_history.history[loss])
    plt.title('Train History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    
show_train_history('loss')


# In[25]:


train_predict = model.predict(trainX)
test_predict = model.predict(testX)


# In[26]:


trainPredict = minmax.inverse_transform(train_predict).astype('int64')
trainY = minmax.inverse_transform(trainY.reshape(-1,1))
testPredict = minmax.inverse_transform(test_predict).astype('int64')
testY = minmax.inverse_transform(testY.reshape(-1,1))


# In[27]:


trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:, 0]))
print('Train Score: ', trainScore)
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:, 0]))
print('Test Score: ', testScore)


# In[28]:


plt.figure(figsize=(10,5))
plt.plot(trainY.reshape(-1)[450:], color = 'blue', linewidth = 3)
plt.plot(trainPredict.reshape(-1)[450:], color = 'red', linewidth = 3)
plt.title('Training Set', fontsize=25)
plt.legend(['Real New Cases', 'Predicted New Cases'], loc = 'upper left', fontsize=15)
plt.show


# In[29]:


test_date = date[train_size+1:len(date)]
plt.figure(figsize=(10,5))
plt.plot(test_date, testY.reshape(-1), color = 'blue', linewidth = 3)
plt.plot(test_date, testPredict.reshape(-1), color = 'red', linewidth = 3)
plt.title('Training Set', fontsize=25)
plt.legend(['Real New Cases', 'Predicted New Cases'], loc = 'upper right', fontsize=15)
plt.xlabel("date",fontsize=18)
plt.ylabel("New Cases",fontsize=18)
plt.show


# In[30]:


testY


# In[31]:


testPredict


# In[32]:


futurePredict = testY[-1:]
def predict_newCases(val=futurePredict[-1:]):
    global futurePredict
    val_minmax = minmax.transform(val.reshape(-1,1))
    future_predict = model.predict(val_minmax.reshape(-1,1,1))
    future_Predict = minmax.inverse_transform(future_predict)
    future_Predict = np.array(future_Predict, dtype = 'int64')
    futurePredict = np.append(futurePredict, future_Predict, axis = 0)
    return futurePredict
predict_newCases()


# In[ ]:




