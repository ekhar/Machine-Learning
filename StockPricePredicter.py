import pandas as pd
import quandl, math
import numpy as np
import pandas_datareader.data as web
import datetime
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import time

style.use('ggplot') #Makes graph look nice

start_day = datetime.date(2013,1,1)
end_day = datetime.date(2018,4,10)

delta_days = end_day - start_day

days_inbetween = [start_day]

for i in range(delta_days.days + 1):
    days_inbetween.append(start_day + datetime.timedelta(days=i) )

df = web.get_data_morningstar('AAPL', start_day, end_day)
# df = quandl.get("EOD/HD", authtoken="jMntsFGxviSof9LiXbG2")

df = df[['Open','High','Low','Close','Volume']] #Grab important datasets only

old_days_inbetween = list(df.index) #Get all dates in same format
dictionary_days = dict(zip(old_days_inbetween,days_inbetween))
df.rename(index=dictionary_days, inplace=True)
df.reset_index(inplace=True)
df.drop("Symbol",axis=1,inplace=True)
df.set_index('Date', inplace=True) #Replaces index of Symbol with Date

df['HL_PCT'] = 100 * ((df['High'] - df['Low'])/ df['High']) #How volitile the market is
df['PCT_Change'] = 100 * ((df['Close'] - df['Open'])/ df['Open']) #Did stock go up or down in a given day

forcast_col = 'Close'
df.fillna(-.9999, inplace=True) #Choose -.9999 because it is an outliar
forcast_out = int(math.ceil(.01*len(df))) #To predict 1% of data fram (days) into the future

df['label'] = df[forcast_col].shift(-forcast_out) #Gets a prediction table set
df.dropna(inplace=True)
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X) #for faster training
X_lately = X[-forcast_out:]
X = X[:-forcast_out]
y = np.array(df['label'])
y = y[:-forcast_out]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)

clf = LinearRegression()
# clf = svm.SVR() ##This is the worse one of the 2
clf.fit(X_train, y_train)
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f) #So that we don't have to train every time

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan
last_date = df.iloc[-2].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for variable in range(len(df.columns)-1)]+[i]

df.rename(columns={'Close':'Historic Stock Price', 'Forecast': 'Predicted Stock Price'}, inplace=True)
df['Historic Stock Price'].plot()
df['Predicted Stock Price'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.show()

