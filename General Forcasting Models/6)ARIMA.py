import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel, ylabel
from numpy.core.defchararray import title

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


my_data = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/TradeInventories.csv',
                      index_col=0,parse_dates=True)
print(my_data.head())


# my_data = pd.read_csv("C:/Users/COMTECH COMPUTER/Desktop/weekly pre-dispatch forecast.csv",
#                    index_col=0,parse_dates=True)
# print(my_data.head())

# # Converting Duplicate values into single one
# my_data = my_data.groupby('datetime').sum()
# my_data.plot(figsize=(12,8))         #-> Original plot
#
# # my_data = my_data.resample(rule='D').sum()
# # my_data.plot(figsize = (12,8))     #-> Day wise plot
#
# my_data = my_data.resample(rule='MS').sum()
# print(my_data['Inventories'].head())
# my_data.plot(figsize=(12,8))         #->Month wise plot
# plt.show()

#-> plotting of data
title   = 'Real Manufacturing and Trade Inventories'
y_lable = 'Chained 2012 Dollars'
x_lable = ''

data = my_data['Inventories'].plot(figsize=(12,8),title=title)
data.set(xlabel=x_lable,ylabel = y_lable)

# # model='add' also works
result = seasonal_decompose(my_data['Inventories'], model='additive')
result.plot()
plt.show()

"""Use pmdarima auto_arima to determine ARIMA Orders(p,d,q)"""
from pmdarima import auto_arima
model01 = auto_arima(my_data['Inventories'],seasonal=False)
print(model01.summary())
#-> order is (3,0,1)

# Run the augmented Dickey-Fuller Test on the First Difference
from statsmodels.tsa.stattools import adfuller
def adf_test(series, title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    # dropna() handles differenced data
    result = adfuller(series.dropna(), autolag='AIC')

    labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    out = pd.Series(result[0:4], index=labels)

    for key, val in result[4].items():
        out[f'critical value ({key})'] = val

    print(out.to_string())  # .to_string() removes the line "dtype: float64"

    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

from statsmodels.tsa.statespace.tools import diff
#-> differencing the data by 1 lag
my_data['d1'] = diff(my_data['Inventories'],k_diff=2)
# Equivalent to:
# df1['d1'] = df1['load_forecast'] - df1['load_forecast'].shift(1)

adf_test(my_data['d1'],'Real Manufacturing and Trade Inventories')
print(my_data.head())

"""Run the ACF and PACF plots
# A PACF Plot can reveal recommended AR(p) orders, and an ACF Plot can do the same for MA(q) orders.
# Alternatively, we can compare the stepwise Akaike Information Criterion (AIC) values across a set of different (p,q) combinations to choose the best combination."""

# ACF
title = 'Autocorrelation: Real Manufacturing and Trade Inventories'
lags = 40
plot_acf(my_data['Inventories'],title=title,lags=lags)

# PACF
title = 'Partial Autocorrelation: Real Manufacturing and Trade Inventories'
lags = 25
plot_pacf(my_data['Inventories'],title=title,lags=lags)
plt.show()

stepwise_fit = auto_arima(my_data['Inventories'], start_p=0, start_q=0,
                          max_p=3, max_q=2, m=12,
                          seasonal=False,
                          d=None, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise

print(stepwise_fit.summary())

# Split the data into train/test sets
# Set one year for testing
train = my_data.iloc[:252]
test = my_data.iloc[252:]

"""Fit an ARIMA(1,1,1) Model"""
model = ARIMA(train['Inventories'],order=(3,0,1))
results = model.fit()
print(results.summary())

# Obtain predicted values(Forcasting)
start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end)
predictions.rename('ARIMA(1,1,1) Predictions')

# # Compare predictions to expected values
# for i in range(len(predictions)):
#     print(f"predicted={predictions[i]:<11.10}, expected={test['load_forecast'][i]}")

# Plot predictions against known values
title = 'Real Manufacturing and Trade Inventories'
ylabel='Chained 2012 Dollars'
xlabel=''

ax = test['Inventories'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
plt.show()


"""Evaluate the Mode(checking the Error)"""
# sme,rmse
# Not correct
# from sklearn.metrics import mean_squared_error
# error = mean_squared_error(test['Inventories'], predictions)
# print(f'ARIMA(1,1,1) MSE Error: {error:11.10}')
#
# from statsmodels.tools.eval_measures import rmse
# error = rmse(test['Inventories'], predictions)
# print(f'ARIMA(1,1,1) RMSE Error: {error:11.10}')


"""etrain the model on the full data, and forecast the future"""
model = ARIMA(my_data['Inventories'],order=(3,0,1))
results = model.fit()
fcast = results.predict(start=len(my_data),end=len(my_data)+11,typ='levels').rename('ARIMA(1,1,1) Forecast')

# Plot predictions against known values
title = 'Real Manufacturing and Trade Inventories'
ylabel='Chained 2012 Dollars'
xlabel='' # we don't really need a label here

ax = my_data['Inventories'].plot(legend=True,figsize=(12,6),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
plt.show()
