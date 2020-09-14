import quandl
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt

from sklearn.neighbors import KNeighborsClassifier

style.use('ggplot')

df = pd.read_csv("C:\\Users\\kassa\\stock prices\\GOOG.csv",index_col='Date', parse_dates=True)
df = df[['Adj Close']]

forecast_out = 5

df['prediction'] = df['Adj Close'].shift(-forecast_out)

X = np.array(df.drop(['prediction'], 1))
X_forecast = X[-forecast_out:]
X = X[:-forecast_out]
copy = df.dropna()
y = np.array(copy.drop(['Adj Close'], 1))

y = y.tolist()


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

clf = LinearRegression()
# clf = KNeighborsClassifier()

clf.fit(X_train , y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
forecast_set = clf.predict(X_forecast)
final_col = y

for i in forecast_set:
    final_col.append(i)

for i in range(len(final_col)):
    final_col[i] = final_col[i][0]
df.fillna(0, inplace = True)
df['prediction'] = final_col

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in final_col[-forecast_out:]:
    next_date = dt.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] +[i]
# df.fillna(0, inplace = True)
print(df)
df['Adj Close'].plot()
df['prediction'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()