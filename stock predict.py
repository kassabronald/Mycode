import numpy as np
# to create 2D arrays processed
import pandas as pd
# to read csv_file
from sklearn.model_selection import train_test_split
# split our data between training and testing data
from sklearn.linear_model import LinearRegression
# Import the Linear regression method
import matplotlib.pyplot as plt
# plot the results
from matplotlib import style
import datetime as dt

style.use('ggplot')
# just a conventional style to use for stock showing
def mean_by_hand(array):
    mean = 0
    for i in range(len(array)):
        mean += array[i][0]
    return mean//len(array)

#function we will use later on


df = pd.read_csv("C:\\Users\\kassa\\stock prices\\GOOG.csv", index_col='Date', parse_dates=True)
# read the data (download from yahoo finance and save as csv)
df = df[['Adj Close']]
# we only need the adjusted close column as it is the most relevant for stock price but feel free to change it

forecast_out = 5
# number of days we want to 'predict'

df['prediction'] = df['Adj Close'].shift(-forecast_out)
# creates the prediction column and assign to it prices in the future using the shift methid

X = np.array(df.drop(['prediction'], 1))
# Basically the adjusted close prices
X_forecast = X[-forecast_out:]
# Basically the (forecast_out) number of days we want to predict
X = X[:-forecast_out]
# we drop the lattest
copy = df.dropna()
# created a copy in order so i don't modify the initial dataframe
y = np.array(copy.drop(['Adj Close'], 1))
# basically our prediction values

y = y.tolist()
# turn it into a list to diminish confusion in the next step


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# What we are saying here is that we want 25% of our data to be used for testing. Data will be selected at random.

method = LinearRegression()
#We use linear regression here. Dependent Variable is the Price and independent Variable is the date. Obviously this is a straightforward algorithm, stock prediction is much more complex than that.

method.fit(X_train, y_train)
# Basically what is happening is that using those 2 datasets which are our training data, we compute a Linear regression model.
#In other words, we compute the best_fit slope as well as the y-intercept, b=mean(y)-m*mean(x)
def get_m_and_get_b_by_hand(Training_data, target_values):
    x_times_y = 1
    x_times_x = 1
    for i in range(len(Training_data)):
        x_times_y += Training_data[i][0] * target_values[i][0]
        x_times_x += Training_data[i][0] * Training_data[i][0]
    m = (mean_by_hand(Training_data)*mean_by_hand(target_values) - (x_times_y/len(Training_data))) / (mean_by_hand(Training_data)**2 - (x_times_x)/len(Training_data))
    b = mean_by_hand(target_values) - m*mean_by_hand(Training_data)
    return m, b


accuracy = method.score(X_test, y_test)
#We compute the Coefficient of Determination R^2.

def coeff_determination_by_hand(True_values, Predicted_values):
    sum_1 = 0
    sum_2 = 0
    for i in range(len(X_test)): #don't forget X_test and y_test have the same lenght = len(X)/4
        sum_1 += (Predicted_values[i][0] - True_values[i][0])**2
        sum_2 += (True_values[i][0] - mean_by_hand(True_values))**2
    R =  (1 - (sum_1/sum_2))**0.5
    return R*R

#Do it by hand because why not !

m, b = get_m_and_get_b_by_hand(X_train, y_train)
def predict_values_by_hand(Values_to_test):
    forecast = []
    for i in Values_to_test:
        forecast.append(m * i[0] + b )
    return forecast

forecast_set = method.predict(X_forecast)



#simply predict the values by plugging in the X_forecast values in our equation of the form y = mx + b
final_col = y

forecast = predict_values_by_hand(X_forecast)




for i in forecast:  #change_to_forecast_set to do it with built_in functions
    final_col.append([i])

print(final_col)


for i in range(len(final_col)):
    final_col[i] = final_col[i][0]
df.fillna(0, inplace=True)
df['prediction'] = final_col



last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in final_col[-forecast_out:]:
    next_date = dt.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
#set the date for the predicted values



print(df)
df['Adj Close'].plot()
df['prediction'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#plot it :)
