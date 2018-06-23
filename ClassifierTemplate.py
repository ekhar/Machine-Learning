# -*- coding: utf-8 -*-
"""
Created on Tue May 15 21:32:27 2018

@author: Eric
"""

#Logistic Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
data_set = pd.read_csv("Social_Network_Ads.csv")
X = data_set.iloc[:,2:4].values
y = data_set.iloc[:,4].values

#Encoding categorial data
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0]) #Changes "France", "Germany" into 1,2
onehotencoder = OneHotEncoder(categorical_features = [0]) #Selects x collumn
X = onehotencoder.fit_transform(X).toarray() #Changes 1,2 into 1,0 and 0,1
y = labelencoder.fit_transform(y) #Yes,No = 1,0'''

#Train Test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

#Scaling (most of time not needed)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:,:] = sc_X.fit_transform(X_train[:,:])
X_test[:,:] = sc_X.transform(X_test[:,:])

#Fitting Classifier to Training Test
from sklearn.svm import SVC
classifier = SVC(kernal = 'rbf', random_state=0)
classifier.fit(X_train, y_train)

#Predicting Test Set Results
y_pred = classifier.predict(X_test)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Graph
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
 