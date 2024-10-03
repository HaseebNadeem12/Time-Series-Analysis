import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load specific forecasting tools
# Use ARIMA for ARMA/ARIMA
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# for determining ARIMA orders
from pmdarima import auto_arima

# Load datasets
df1 = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/DailyTotalFemaleBirths.csv',
                  index_col='Date', parse_dates=True)
df1.index.freq = 'D'
# we only want the first four months
df1 = df1[:120]

df2 = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/TradeInventories.csv',
                  index_col='Date', parse_dates=True)
df2.index.freq = 'MS'

from statsmodels.tsa.stattools import adfuller
#-> Dicky Fullar test(tells about stationary or non-stationary data)
def adf_test(series, title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(), autolag='AIC')  # .dropna() handles differenced data

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

"""
Autoregressive Moving Average - ARMA(p,q)
In this first section we'll look at a stationary dataset, 
determine (p,q) orders, and run a forecasting ARMA model 
fit to the data. In practice it's rare to find stationary 
data with no trend or seasonal component, 
but the first four months of the 
Daily Total Female Births dataset should work for our purposes."""

df1['Births'].plot(figsize=(12,5))
plt.show()

# checking data is statioary or not with the help of dicky fuller test
print(adf_test(df1['Births']))

#-> tells us about (p,q)
auto_prem = auto_arima(df1['Births'], seasonal=False)
print(auto_prem.summary())

"""ARMA model Forecasting"""
train = df1[:90]
test = df1[90:]

# Training part of the model using ARIMA with d=0 (this makes it ARMA)
# ARIMA with d=0 to make it ARMA(2,2)
model = ARIMA(train['Births'], order=(2, 0, 2))
results = model.fit()
print(results.summary())

# Prediction part
start = len(train)
end = len(train) + len(test) - 1
predictions = results.predict(start=start, end=end).rename('ARMA(2,2) Predictions')
print(predictions)

# Plotting
title = 'Daily Total Female Births'
ylabel='Births'
xlabel='' # we don't really need a label here

ax = test['Births'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
plt.show()


