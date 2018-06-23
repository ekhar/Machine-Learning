#Regression Template
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
data_set = pd.read_csv("")
X = data_set.iloc[:,1:2].values
y = data_set.iloc[:,2].values

'''#Train Test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)'''

'''Scaling (most of time not needed)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''

#Fitting the Regression Model to dataset
#Create Regressor

#Predict Result with PolyReg
y_pred = regressor.predict(10)

#Visualize Reg Results
plt.scatter(X,y, color = 'red')
plt.plot(X, regressor.predict(X),color = 'blue')
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.show()

#Visualize Reg Results (higher resolution and smoother curve)
X_grid = np.arange(min(X),max(X),.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid),color = 'blue')
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.show()

