"""  VARMA """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load specific forecasting tools
from statsmodels.tsa.api import VAR,VARMAX
# from statsmodels.tsa.vector_ar.dynamic import DynamicVAR  # Corrected import
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA #,ARMAResults
from statsmodels.tools.eval_measures import rmse

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

# Load datasets
df = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/M2SLMoneyStock.csv',index_col=0, parse_dates=True)
df.index.freq = 'MS'

sp = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/PCEPersonalSpending.csv',index_col=0, parse_dates=True)
sp.index.freq = 'MS'

#-> merging both datasets into one
df = df.join(sp)
print(df.head())

df = df.dropna()
print(df.shape)

"""Plotting the data"""
title = 'M2 Money Stock vs. Personal Consumption Expenditures'
ylabel='Billions of dollars'
xlabel=''

ax = df['Spending'].plot(figsize=(12,5),title=title,legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
df['Money'].plot(legend=True)
plt.show()

"""Test for stationarity, perform any necessary transformations"""


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

"""Model Selection"""
#-> As VARMA model support auto_arima, so we can use it to find order

auto_arima(df['Money'],maxiter=1000)
auto_arima(df['Spending'],maxiter=1000)

#-> As VARMA model not support differencing (d) term, so we have to calculate it sepratly

# It looks like a VARMA(1,2) model is recommended. Note that the
#  term (2 for Money, 1 for Spending) is about to be addressed by transforming
#  the data to make it stationary. As before we'll apply a second order difference.

df_transformed = df.diff().diff()
df_transformed = df_transformed.dropna()
print(df_transformed.head())

"""Train/test split"""
#-> It will be useful to define a number of observations variable for our test set. For this analysis, let's use 12 months.
nobs=12
train, test = df_transformed[0:-nobs], df_transformed[-nobs:]
print(train.shape)
print(test.shape)


"""Fit the VARMA(5) Model"""
model = VARMAX(train, order=(1,2), trend='c')
results = model.fit(maxiter=1000, disp=False)
results.summary()

"""Predict the next 12 values"""
# Unlike the VAR model we used in the previous section, the VARMAX
# .forecast() function won't require that we pass in a number of
# previous observations, and it will provide an extended DateTime index
df_forecast = results.forecast(12)
print(df_forecast)

"""Invert the Transformation"""
"""
Remember that the forecasted values represent second-order differences.
To compare them to the original data we have to roll back each difference.
To roll back a first-order difference we take the most recent value on the
training side of the original series, and add it to a cumulative sum of
forecasted values. When working with second-order differences we first must
perform this operation on the most recent first-order difference.

Here we'll use the nobs variable we defined during the train/test/split step.
"""

# Add the most recent first difference from the training side of the original dataset to the forecast cumulative sum
df_forecast['Money1d'] = (df['Money'].iloc[-nobs-1]-df['Money'].iloc[-nobs-2]) + df_forecast['Money'].cumsum()

# Now build the forecast values from the first difference set
df_forecast['MoneyForecast'] = df['Money'].iloc[-nobs-1] + df_forecast['Money'].cumsum()

# Add the most recent first difference from the training side of the original dataset to the forecast cumulative sum
df_forecast['Spending1d'] = (df['Spending'].iloc[-nobs-1]-df['Spending'].iloc[-nobs-2]) + df_forecast['Spending'].cumsum()

# Now build the forecast values from the first difference set
df_forecast['SpendingForecast'] = df['Spending'].iloc[-nobs-1] + df_forecast['Spending'].cumsum()

print(df_forecast)
print(df_forecast.columns)

value = pd.concat([df.iloc[-12:],df_forecast[['MoneyForecast','SpendingForecast']]],axis=1)
print(value)

"""Plot the results"""
#-> But for our investigation we want to plot predicted values against our test set.
df['Money'][-nobs:].plot(figsize=(12,5),legend=True).autoscale(axis='x',tight=True)
df_forecast['MoneyForecast'].plot(legend=True)

df['Spending'][-nobs:].plot(figsize=(12,5),legend=True).autoscale(axis='x',tight=True)
df_forecast['SpendingForecast'].plot(legend=True)
plt.show()

"""Evaluate the model"""
RMSE1 = rmse(df['Money'][-nobs:], df_forecast['MoneyForecast'])
print(f'Money VAR(5) RMSE: {RMSE1:.3f}')
#->Money VARMA(5) RMSE: 422.942

RMSE2 = rmse(df['Spending'][-nobs:], df_forecast['SpendingForecast'])
print(f'Spending VAR(5) RMSE: {RMSE2:.3f}')
#->Spending VARMA(5) RMSE: 243.777


"""Let's compare these results to individual AR(5) models"""
from statsmodels.tsa.ar_model import AR,ARResults

"""Money"""
model = ARIMA(train['Money'],order=(1,0,2))
results = model.fit()
print(results.summary())

start=len(train)
end=len(train)+len(test)-1
z1 = results.predict(start=start, end=end).rename('Money')
z1 = pd.DataFrame(z1)
print(z1)


"""Invert the Transformation, Evaluate the Forecast"""
# Add the most recent first difference from the training set to the forecast cumulative sum
z1['Money1d'] = (df['Money'].iloc[-nobs-1]-df['Money'].iloc[-nobs-2]) + z1['Money'].cumsum()

# Now build the forecast values from the first difference set
z1['MoneyForecast'] = df['Money'].iloc[-nobs-1] + z1['Money1d'].cumsum()
print(z1)

RMSE3 = rmse(df['Money'][-nobs:], z1['MoneyForecast'])

print(f'Money VARMA(1,2) RMSE: {RMSE1:.3f}')
print(f'Money  ARMA(1,2) RMSE: {RMSE3:.3f}')
# Money VAR(5) RMSE: 422.710
# Money  AR(5) RMSE: 32.222


"""Personal Spending"""
model = ARIMA(train['Spending'],order=(1,0,2))
results = model.fit()
results.summary()

start=len(train)
end=len(train)+len(test)-1
z2 = results.predict(start=start, end=end).rename('Spending')
z2 = pd.DataFrame(z2)
print(z2)


"""Invert the Transformation, Evaluate the Forecast"""
# Add the most recent first difference from the training set to the forecast cumulative sum
z2['Spending1d'] = (df['Spending'].iloc[-nobs-1]-df['Spending'].iloc[-nobs-2]) + z2['Spending'].cumsum()

# Now build the forecast values from the first difference set
z2['SpendingForecast'] = df['Spending'].iloc[-nobs-1] + z2['Spending1d'].cumsum()
print(z2)

RMSE4 = rmse(df['Spending'][-nobs:], z2['SpendingForecast'])

print(f'Spending VARMA(1,2) RMSE: {RMSE2:.3f}')
print(f'Spending  ARMA(1,2) RMSE: {RMSE4:.3f}')
# Spending VARMA(1,2) RMSE: 243.777
# Spending  ARMA(1,2) RMSE: 52.334

# CONCLUSION: It looks like the VARMA(1,2) model did a relatively poor job
# compared to simpler alternatives. This tells us that there is little or
# no interdepence between Money Stock and Personal Consumption Expenditures,
# at least for the timespan we investigated. This is helpful! By fitting
# a model and getting poor results we know more about the data than we did before.




