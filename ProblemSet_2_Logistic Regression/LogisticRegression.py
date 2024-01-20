#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings


# # Read Data

# In[2]:


df = pd.read_csv("bank-full.csv", sep = ";")


# In[3]:


df


# In[4]:


df.info()


# # Label Encoding On Data

# In[5]:


catagorical_columns = df.dtypes[df.dtypes == 'object']
numerical_columns = df.dtypes[df.dtypes != 'object']
print(catagorical_columns)
label_encoder = LabelEncoder()
for column in catagorical_columns.index:
    df[column] = label_encoder.fit_transform(df[column])


# # Define X & y

# In[6]:


X = df.iloc[:,0:-1]
y = df.iloc[:,-1]


# In[7]:


X


# In[8]:


y


# # Train_Test_Split

# In[9]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[10]:


X_train


# In[11]:


X_test


# In[12]:


y_train


# In[13]:


y_test


# # Logistic Regression Using Sklearn

# In[14]:


model = LogisticRegression(max_iter=5000)


# In[15]:


model.fit(X_train,y_train)


# In[16]:


y_pred = model.predict(X_test)


# In[17]:


confusion_matrix(y_test, y_pred)


# In[18]:


accuracy_score(y_test,y_pred)


# In[20]:


print(classification_report(y_test, y_pred))


# # Logistic Regression Using Gradient Descent

# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[22]:


class LogisticRegressionGD:
    def __init__(self, learningRate=0.01,num_it = 5000):
        self.learningRate = learningRate
        self.num_it = num_it
        self.features = None
        self.bias = None

    def sigmoid(self, z):
        warnings.filterwarnings('ignore')
        return 1/(1+np.exp(-z))

    def fitGD(self,X,y):
        num_samples,num_features = X.shape
        self.features = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_it):
            linear_model = np.dot(X,self.features) + self.bias
            y_pred = self.sigmoid(linear_model)

            w = (1/num_samples)*np.dot(X.T,(y_pred-y))
            b = (1/num_samples)*np.sum(y_pred-y)
            self.features -= self.learningRate*w
            self.bias -= self.learningRate*b

    def predict(self,X):
        linear_model = np.dot(X,self.features) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_cls = [1 if i>0.5 else 0 for i in y_pred]
        return np.array(y_pred_cls)

    def accuracy(self, y_test,y_pred):
        accuracy = np.sum(y_test == y_pred)/len(y_test)
        return accuracy


# In[23]:


model = LogisticRegressionGD(learningRate=0.01,num_it=5000)


# In[24]:


model.fitGD(X_train,y_train)


# In[25]:


y_pred = model.predict(X_test)


# In[26]:


confusion_matrix(y_test,y_pred)


# In[27]:


model.accuracy(y_test,y_pred)


# In[28]:


print(classification_report(y_test,y_pred))


# In[ ]:




