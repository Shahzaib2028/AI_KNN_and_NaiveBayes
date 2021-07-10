#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, scipy, numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

ds = pd.read_csv('tennis.csv', usecols= ['outlook','temp','humidity','windy','play'])
# inspect some basic attributes of the data set
ds.info()
x = ds.iloc[:,:-1].values
y = ds.iloc[:,-1].values
print(x)
print(y)

encoder = LabelEncoder()

#Converting String Labels into numbers one column at a time
x[:,0]=encoder.fit_transform(x[:,0])
x[:,1] = encoder.fit_transform(x[:,1])
x[:,2] = encoder.fit_transform(x[:,2])
x[:,3] = encoder.fit_transform(x[:,3])
y = encoder.fit_transform(y)

#Observing the transformed data set
print("Outlook:", x[:,0])
print("Temp:", x[:,1])
print("Humidity:", x[:,2])
print("Windy:", x[:,3])
print("Play:",y)

#Create a Multinomail Naive Bayes Classifier
model = MultinomialNB()

#Train the model
model.fit(x,y)

#predict Output
predicted = model.predict([[2,2,1,0]]) #input coming as overcast and Mild
print("Predicted Value:", predicted)


# In[ ]:




