#!/usr/bin/env python
# coding: utf-8

# In[17]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV#自動調參
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_roc_curve
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('D:\IMLP342\data\\titanic_data.csv')
df.head()


# In[3]:


trainX, testX, trainY, testY = train_test_split(
    df.drop('Survived', axis = 1), df['Survived'], test_size = 0.3, random_state = 1)
trainY = pd.DataFrame(trainY)
testY = pd.DataFrame(testY)


# In[4]:


trainX


# In[5]:


testX


# In[6]:


pd.isnull(df).sum()


# In[7]:


trainX['Age'].fillna(trainX['Age'].median(), inplace = True)
testX['Age'].fillna(testX['Age'].median(), inplace = True)
trainX['Embarked'].fillna('S', inplace = True)
testX['Embarked'].fillna('S', inplace = True)

def get_nulls(training, testing):
    print("Training Data:")
    print(pd.isnull(training).sum())
    print("Testing Data:")
    print(pd.isnull(testing).sum())
get_nulls(trainX, testX)


# In[8]:


encoder_1 = LabelEncoder()
trainX_sex_encod = encoder_1.fit_transform(trainX["Sex"])
trainX['Sex'] = trainX_sex_encod
testX_sex_encod = encoder_1.transform(testX["Sex"])
testX['Sex'] = testX_sex_encod

encoder_2 = LabelEncoder()
trainX_embarked_encod = encoder_2.fit_transform(trainX["Embarked"])
trainX["Embarked"] = trainX_embarked_encod
testX_embarked_encod = encoder_2.transform(testX["Embarked"])
testX["Embarked"] = testX_embarked_encod

encoder_3 = LabelEncoder()
trainY_encod = encoder_3.fit_transform(trainY["Survived"])
trainY["Survived"] = trainY_encod
testY_encod = encoder_3.transform(testY["Survived"])
testY["Survived"] = testY_encod

ages_train = np.array(trainX["Age"]).reshape(-1, 1)
fares_train = np.array(trainX["Fare"]).reshape(-1, 1)
ages_test = np.array(testX["Age"]).reshape(-1, 1)
fares_test = np.array(testX["Fare"]).reshape(-1, 1)

scaler = StandardScaler()

trainX["Age"] = scaler.fit_transform(ages_train)
trainX["Fare"] = scaler.fit_transform(fares_train)
testX["Age"] = scaler.transform(ages_test)
testX["Fare"] = scaler.transform(fares_test)


# In[9]:


trainX.head()


# In[10]:


# 建立 random forest 模型
forest = RandomForestClassifier(n_estimators = 100)
forest_fit = forest.fit(trainX, trainY)


# In[11]:


# 預測
testY_predicted = forest.predict(testX)
testY_predicted


# In[12]:


# 績效
accuracy = accuracy_score(testY, testY_predicted)
print(accuracy)


# In[18]:


ax = plt.gca()
rfc_disp = plot_roc_curve(forest, testX, testY, ax=ax, alpha=0.8)
plt.show()

