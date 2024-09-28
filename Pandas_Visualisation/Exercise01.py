import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde  # Ensure scipy is installed

# Reading CSV files
data3 = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/Data/Pandas Visualisation/df3.csv')
print(data3.head())

"""1. Recreate this scatter plot of 'produced' vs 'defective'."""
# data3.plot.scatter(x='produced',y='defective',figsize=(12,5),c='red')

"""2. Create a histogram of the 'produced' column."""
# print(data3['produced'].hist())

"""3. Recreate a histogram of the 'produced' column. with boarder"""
# print(data3['produced'].hist(edgecolor = 'k'))

"""4. Create a boxplot that shows 'produced' for each 'weekday'"""
# print(data3['produced'].groupby('weekday').plot.box())

"""5. Create a KDE plot of the 'defective' column"""
# print(data3['defective'].plot.kde())

"""6. For the above KDE plot, figure out how to increase the linewidth and make the linestyle dashed."""
print(data3['defective'].plot.kde(ls='-.',lw=4 ,figsize= (12,5),c='blue'))

"""7. Create a blended area plot of all the columns for just the rows up to 30. (hint: use .loc)"""
print(data3.plot.area(figsize=(12,5)))

plt.show()