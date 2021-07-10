#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the Libraries
import pandas as pd, scipy, numpy as np
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#Loading the iris dataset
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class'] #Assigning the headers
ds = pd.read_csv('iris.csv', names = headernames)
ds.head()

#Splitting up in feature attributes and class variable
x = ds.iloc[:, :-1].values
print(x)

y=ds.iloc[:, 4].values
print(y)

#Train and Test Split
#Next, we will divide the data into train and test split. 
#Following code will split the dataset into 60% training data and 40% of testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.40)

#Data Scaling using the StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Training a KNN Classifier
classifier = KNeighborsClassifier(n_neighbors = 7)
classifier.fit(X_train, y_train)

#Making the Predictions
y_pred = classifier.predict(X_test)

#OUTPUT Metrics
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


# In[ ]:




