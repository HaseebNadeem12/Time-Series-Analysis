"""  VAR  """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load specific forecasting tools
from statsmodels.tsa.api import VAR
# from statsmodels.tsa.vector_ar.dynamic import DynamicVAR  # Corrected import
from statsmodels.tsa.stattools import adfuller
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

adf_test(df['Money'],title='Money')
adf_test(df['Spending'], title='Spending')

#-> Trying to make data stationary with the help of differencing
#-> 1st difference
df_transformed = df.diff()
df_transformed = df_transformed.dropna()

adf_test(df_transformed['Money'], title='MoneyFirstDiff')
adf_test(df_transformed['Spending'], title='SpendingFirstDiff')

#-> 2ND difference
df_transformed = df_transformed.diff().dropna()

adf_test(df_transformed['Money'], title='MoneySecondDiff')
adf_test(df_transformed['Spending'], title='SpendingSecondDiff')

print(df.head())

"""Train/test split"""
#-> It will be useful to define a number of observations variable for our test set. For this analysis, let's use 12 months.
nobs=12
train, test = df_transformed[0:-nobs], df_transformed[-nobs:]
print(train.shape)
print(test.shape)

"""VAR Model Order Selection"""
#-> As VAR model does not support Auto-arima so we have to find order with the help of loop
#-> We'll fit a series of models using the first seven p-values, and base our final selection on the model that provides the lowest AIC and BIC scores.

for i in [1,2,3,4,5,6,7]:  #can also use range(8)
    model = VAR(train)
    results = model.fit(i)
    print('Order =', i)
    print('AIC: ', results.aic)
    print('BIC: ', results.bic)
    print()

model = VAR(train)
for i in [1,2,3,4,5,6,7]:
    results = model.fit(i)
    print('Order =', i)
    print('AIC: ', results.aic)
    print('BIC: ', results.bic)
    print()

print(model.endog_names)


"""Fit the VAR(5) Model"""
results = model.fit(5)
results.summary()

"""Predict the next 12 values"""
#Unlike the VARMAX model we'll use in upcoming sections, the VAR .forecast()
# function requires that we pass in a lag order number of previous observations
# as well. Unfortunately this forecast tool doesn't provide a DateTime
# index - we'll have to do that manually.

lag_order = results.k_ar
print(lag_order)

z = results.forecast(y=train.values[-lag_order:], steps=12)
print(z)
print(test)

idx = pd.date_range('1/1/2015', periods=12, freq='MS')
df_forecast = pd.DataFrame(z, index=idx, columns=['Money2d','Spending2d'])
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
df_forecast['Money1d'] = (df['Money'].iloc[-nobs-1]-df['Money'].iloc[-nobs-2]) + df_forecast['Money2d'].cumsum()

# Now build the forecast values from the first difference set
df_forecast['MoneyForecast'] = df['Money'].iloc[-nobs-1] + df_forecast['Money1d'].cumsum()

# Add the most recent first difference from the training side of the original dataset to the forecast cumulative sum
df_forecast['Spending1d'] = (df['Spending'].iloc[-nobs-1]-df['Spending'].iloc[-nobs-2]) + df_forecast['Spending2d'].cumsum()

# Now build the forecast values from the first difference set
df_forecast['SpendingForecast'] = df['Spending'].iloc[-nobs-1] + df_forecast['Spending1d'].cumsum()

print(df_forecast)
print(df_forecast.columns)

"""Plot the results"""
#-> The VARResults object offers a couple of quick plotting tools:
results.plot()
plt.show()
results.plot_forecast(12)
plt.show()

#-> But for our investigation we want to plot predicted values against our test set.
df['Money'][-nobs:].plot(figsize=(12,5),legend=True).autoscale(axis='x',tight=True)
df_forecast['MoneyForecast'].plot(legend=True)

df['Spending'][-nobs:].plot(figsize=(12,5),legend=True).autoscale(axis='x',tight=True)
df_forecast['SpendingForecast'].plot(legend=True)


"""Evaluate the model"""
RMSE1 = rmse(df['Money'][-nobs:], df_forecast['MoneyForecast'])
print(f'Money VAR(5) RMSE: {RMSE1:.3f}')
#->Money VAR(5) RMSE: 43.710

RMSE2 = rmse(df['Spending'][-nobs:], df_forecast['SpendingForecast'])
print(f'Spending VAR(5) RMSE: {RMSE2:.3f}')
#->Spending VAR(5) RMSE: 37.001


"""Let's compare these results to individual AR(5) models"""
from statsmodels.tsa.ar_model import AR,ARResults
"""Money"""
modelM = AR(train['Money'])
AR5fit1 = modelM.fit(maxlag=5,method='mle')
print(f'Lag: {AR5fit1.k_ar}')
print(f'Coefficients:\n{AR5fit1.params}')

start=len(train)
end=len(train)+len(test)-1
z1 = pd.DataFrame(AR5fit1.predict(start=start, end=end, dynamic=False),columns=['Money'])
print(z1)


"""Invert the Transformation, Evaluate the Forecast"""
# Add the most recent first difference from the training set to the forecast cumulative sum
z1['Money1d'] = (df['Money'].iloc[-nobs-1]-df['Money'].iloc[-nobs-2]) + z1['Money'].cumsum()

# Now build the forecast values from the first difference set
z1['MoneyForecast'] = df['Money'].iloc[-nobs-1] + z1['Money1d'].cumsum()
print(z1)

RMSE3 = rmse(df['Money'][-nobs:], z1['MoneyForecast'])

print(f'Money VAR(5) RMSE: {RMSE1:.3f}')
print(f'Money  AR(5) RMSE: {RMSE3:.3f}')
# Money VAR(5) RMSE: 43.710
# Money  AR(5) RMSE: 36.222


"""Personal Spending"""
modelS = AR(train['Spending'])
AR5fit2 = modelS.fit(maxlag=5,method='mle')
print(f'Lag: {AR5fit2.k_ar}')
print(f'Coefficients:\n{AR5fit2.params}')

z2 = pd.DataFrame(AR5fit2.predict(start=start, end=end, dynamic=False),columns=['Spending'])
print(z2)

"""Invert the Transformation, Evaluate the Forecast"""
# Add the most recent first difference from the training set to the forecast cumulative sum
z2['Spending1d'] = (df['Spending'].iloc[-nobs-1]-df['Spending'].iloc[-nobs-2]) + z2['Spending'].cumsum()

# Now build the forecast values from the first difference set
z2['SpendingForecast'] = df['Spending'].iloc[-nobs-1] + z2['Spending1d'].cumsum()
print(z2)

RMSE4 = rmse(df['Spending'][-nobs:], z2['SpendingForecast'])

print(f'Spending VAR(5) RMSE: {RMSE2:.3f}')
print(f'Spending  AR(5) RMSE: {RMSE4:.3f}')
# Spending VAR(5) RMSE: 37.001
# Spending  AR(5) RMSE: 34.121

# CONCLUSION: It looks like the VAR(5) model did not do better than
# the individual AR(5) models. That's ok - we know more than we did
# before. In the next section we'll look at VARMA and see if the
# addition of a parameter helps. Great work!












