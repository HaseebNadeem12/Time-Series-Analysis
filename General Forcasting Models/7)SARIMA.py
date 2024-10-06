from calendar import month
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load specific forecasting tools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose          # for ETS Plots
from pmdarima import auto_arima                                # for determining ARIMA orders

# Ignore harmless warnings
import warnings

from statsmodels.tsa.vector_ar.var_model import forecast

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/co2_mm_mlo.csv')
print(df.head())

# Add a "date" datetime column
df['date'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=1))
df.set_index('date', inplace=True)  # Corrected this line
df.index.freq = 'MS'
print(df.head())

title = 'Monthly Mean CO₂ Levels (ppm) over Mauna Loa, Hawaii'
ylabel = 'parts per million'
xlabel = ''

# Check if 'interpolated' column exists before plotting
if 'interpolated' in df.columns:
    ax = df['interpolated'].plot(figsize=(12, 6), title=title)
    ax.autoscale(axis='x', tight=True)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.show()
else:
    print("Column 'interpolated' does not exist in the DataFrame.")

result = seasonal_decompose(df['interpolated'], model='add')
result.plot()
plt.show()

# For SARIMA Orders we set seasonal=True and pass in an m value
auto_arima(df['interpolated'],seasonal=True,m=12).summary()
# Set one year for testing
train = df.iloc[:717]
test = df.iloc[717:]

model = SARIMAX(train['interpolated'],order=(0,1,3),seasonal_order=(1,0,1,12))
results = model.fit()
results.summary()

# Obtain predicted values
start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels')
predictions.rename('SARIMA(0,1,3)(1,0,1,12) Predictions')

# Compare predictions to expected values
for i in range(len(predictions)):
    print(f"predicted={predictions[i]:<11.10}, expected={test['interpolated'][i]}")

# Plot predictions against known values
title = 'Monthly Mean CO₂ Levels (ppm) over Mauna Loa, Hawaii'
ylabel='parts per million'
xlabel=''

ax = test['interpolated'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
plt.show()

#-> Evaluation
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

error = mean_squared_error(test['interpolated'], predictions)
print(f'SARIMA(0,1,3)(1,0,1,12) MSE Error: {error:11.10}')

error = rmse(test['interpolated'], predictions)
print(f'SARIMA(0,1,3)(1,0,1,12) RMSE Error: {error:11.10}')


"""Retrain the model on the full data, and forecast the future"""

model = SARIMAX(df['interpolated'],order=(0,1,3),seasonal_order=(1,0,1,12))
results = model.fit()
fcast = results.predict(len(df),len(df)+11,typ='levels')
fcast.rename('SARIMA(0,1,3)(1,0,1,12) Forecast')

# Plot predictions against known values
title = 'Monthly Mean CO₂ Levels (ppm) over Mauna Loa, Hawaii'
ylabel='parts per million'
xlabel=''

ax = df['interpolated'].plot(legend=True,figsize=(12,6),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)



