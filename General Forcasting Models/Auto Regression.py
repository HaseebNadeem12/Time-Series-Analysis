import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load specific forecasting tools
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

# Load the U.S. Population dataset
df = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/uspopulation.csv'
                 ,index_col='DATE',parse_dates=True)
df.index.freq = 'MS'
print(df.head())

title='U.S. Monthly Population Estimates'
ylabel='Pop. Est. (thousands)'
xlabel='' # we don't really need a label here

ax = df['PopEst'].plot(figsize=(12,5),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
plt.show()

"""Forcasting the Model with 1st order"""
# Set one year for testing
train = df.iloc[:84]
test = df.iloc[84:]

model = AutoReg(train['PopEst'],lags=1)
AR1fit = model.fit()
# print(f'Lag: {AR1fit.k_ar}')
print(f'Coefficients:\n{AR1fit.params}')

# This is the general format for obtaining predictions
start=len(train)
end=len(train)+len(test)-1
predictions1 = AR1fit.predict(start=start, end=end, dynamic=False).rename('AR(1) Predictions')
print(predictions1)

# Comparing predictions to expected values
for i in range(len(predictions1)):
    print(f"predicted={predictions1[i]:<11.10}, expected={test['PopEst'][i]}")

test['PopEst'].plot(legend=True)
predictions1.plot(legend=True,figsize=(12,6));
plt.show()

# Recall that our model was already created above based on the training set
model01 = AutoReg(train['PopEst'],lags=2)
AR2fit01 = model01.fit()
# print(f'Lag: {AR2fit.k_ar}')
print(f'Coefficients:\n{AR2fit01.params}')

start=len(train)
end=len(train)+len(test)-1
predictions2 = AR2fit01.predict(start=start, end=end, dynamic=False).rename('AR(2) Predictions')

test['PopEst'].plot(legend=True)
predictions1.plot(legend=True)
predictions2.plot(legend=True,figsize=(12,6))


"""Lets statsmodel to decide what number of lag is best suitaded for forcasting"""
# Use ar_select_order to select the best lag based on AIC
model_selector = ar_select_order(train['PopEst'], maxlag=8, ic='aic', glob=True)

# Get the selected lag value
selected_lag = model_selector.ar_lags[-1]  # Best lag value

# Fit the AutoReg model with the selected lag
ARfit = AutoReg(train['PopEst'], lags=selected_lag).fit()

# Print the selected lag and model coefficients
print(f'Selected Lag: {selected_lag}')
print(ARfit.params)

# Set start and end points for prediction
start = len(train)
end = len(train) + len(test) - 1

# Make predictions using the selected lag
predictions = ARfit.predict(start=start, end=end, dynamic=False).rename(f'AR({selected_lag}) Predictions')

# Plot the predictions against actual values
# test['PopEst'].plot(legend=True)
predictions.plot(legend=True)
plt.show()

from sklearn.metrics import mean_squared_error
labels = [AR1fit,AR2fit01,ARfit]
preds = [predictions1,predictions2,predictions]

for i in range(3):
    result = mean_squared_error(test['PopEst'],preds[i])
    print(i+1 , "RMSE is",np.sqrt(result))


"""Future forcasting"""
model_selector = ar_select_order(train['PopEst'], maxlag=8, ic='aic', glob=True)
selected_lag = model_selector.ar_lags[-1]  # Best lag value
ARfit = AutoReg(train['PopEst'], lags=selected_lag).fit()
forcasted_value = ARfit.predict(start=len(df),end=len(df)+12).rename('forcast')
df['PopEst'].plot()
forcasted_value.plot()
plt.show()
