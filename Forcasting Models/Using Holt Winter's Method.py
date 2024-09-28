import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset
my_file = pd.read_csv("C:/Users/COMTECH COMPUTER/Desktop/UDEMY_TSA_FINAL/Data/airline_passengers.csv",
                      index_col=0, parse_dates=True)

my_file.index.freq='MS'
# Drop any rows with NaN values
# my_file.dropna(inplace=True)

# Display the first 10 rows
print(my_file.head())
my_file['Thousands'].plot()
plt.show()

# Extract feature (X)
X = my_file['Thousands']

# Split the data
X_train = my_file['Thousands'].iloc[:96]
X_test = my_file['Thousands'].iloc[96:]
print(X_test.head())
# X_train.plot()
# plt.show()

# Apply Holt-Winters Exponential Smoothing
hw_model = ExponentialSmoothing(X_train, trend='add', seasonal='mul', seasonal_periods=12).fit()

# Print the fitted model
predictions = hw_model.forecast(5)
print(predictions.head())

# Plot the fitted values (uncomment if needed)
# X_train.plot(label="Original Data")
# predictions.plot(label="Fitted Data", linestyle="--")
# plt.show()
