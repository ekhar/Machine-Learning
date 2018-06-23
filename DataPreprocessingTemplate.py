#Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
data_set = pd.read_csv("")
X = data_set.iloc[:,:-1].values
y = data_set.iloc[:,3].values

#Fill in any missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3]) #Find averages in the collumns 
X[:, 1:3] = imputer.transform(X[:,1:3]) #Change the NaN values

#Encoding categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0]) #Changes "France", "Germany" into 1,2
onehotencoder = OneHotEncoder(categorical_features = [0]) #Selects first collumn
X = onehotencoder.fit_transform(X).toarray() #Changes 1,2 into 1,0 and 0,1
y = labelencoder.fit_transform(y) #Yes,No = 1,0

#Train Test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

'''Scaling (most of time not needed)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''
