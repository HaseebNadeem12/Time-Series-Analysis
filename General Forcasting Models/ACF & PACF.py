import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load a non-stationary dataset
df1 = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/airline_passengers.csv',index_col='Month',parse_dates=True)
df1.index.freq = 'MS'
df1.plot()
plt.show()

# Load a stationary dataset
df2 = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/DailyTotalFemaleBirths.csv',index_col='Date',parse_dates=True)
df2.index.freq = 'D'
df2.plot()
plt.show()

"""Auto-covariance"""
from statsmodels.tsa.stattools import acovf,acf,pacf,pacf_yw,pacf_ols
df = pd.DataFrame({'a':[13, 5, 11, 12, 9]})
arr = acovf(df['a'])
print(arr)

"""Unbiased Auto-covariance"""
# arr2 = acovf(df['a'],unbiased=True)
# print(arr2)

"""Auto-correlation for 1D"""
# arr3 = acf(df['a'])
# print(arr3)

"""Partial Auto-correlation"""
# #->NOTE: We passed in method='mle' above in order to use biased ACF coefficients.
# # "mle" stands for "maximum likelihood estimation".
# # Alternatively we can pass method='unbiased' (the statsmodels default)
# arr4 = pacf_yw(df['a'],nlags=4,method='mle')
# print(arr4)
#
# arr5 = pacf_yw(df['a'],nlags=4,method='unbiased')
# print(arr5)

"""Partial Auto-correlation with OLS(ordinar least square)"""
# arr6 = pacf_ols(df['a'],nlags=4)
# print(arr6)

"""Plotting of ACF and PACF"""
# #-> ACF = auto-corelation function
# #-> PACF-yw = partial auto-corelation function by yowlk worker method
# from pandas.plotting import lag_plot
#
# lag_plot(df1['Thousands']).plot()
# plt.show()
#
# lag_plot(df2['Births']).plot()
# plt.show()

"""ACF Plot"""
# -> consider stationary data
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# Let's look first at the ACF array. By default acf() returns 40 lags
print(acf(df2['Births']))
# Now let's plot the autocorrelation at different lags
title = 'Autocorrelation: Daily Female Births'
lags = 40
plot_acf(df2,title=title,lags=lags)
plt.show()

acf(df1['Thousands'])
title = 'Autocorrelation: Airline Passengers'
lags = 40
plot_acf(df1,title=title,lags=lags)
plt.show()


"""PACF Plots"""
#-> PACF work well with stationary data
title='Partial Autocorrelation: Daily Female Births'
lags=40
plot_pacf(df2,title=title,lags=lags)
plt.show()

from statsmodels.tsa.statespace.tools import diff
#-> first converting non-stationary data into stationary to apply PACF
df1['d1'] = diff(df1['Thousands'],k_diff=1)
df1['d1'].plot(figsize=(12,5))

title='PACF: Airline Passengers First Difference'
lags=40
# be sure to add .dropna() here!
plot_pacf(df1['d1'].dropna(),title=title,lags=np.arange(lags))

fig, ax = plt.subplots(figsize=(12,5))
plot_acf(df1['Thousands'],ax=ax)
plt.show()

