import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde
# Ensure scipy is installed

# Reading CSV files
data1 = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/Data/Pandas Visualisation/df1.csv')
data2 = pd.read_csv('C:/Users/COMTECH COMPUTER/Desktop/Data/Pandas Visualisation/df2.csv')


"""HISTOGRAM"""
# print(data1['A'].plot.hist())                       # without boarder
# print(data1['A'].plot.hist(edgecolor='k'))          # with boarder
# print(data1['A'].hist(edgecolor='k'))               # with grid
# print(data1['A'].plot.hist(bins=30,edgecolor='k'))  # Multiple bins

"""BAR PLOT"""
# print(data2.plot.bar())
# print(data2.plot.bar(stacked=True))
# print(data2.plot.barh())                            # horizontal bars

"""Line plot"""
# print(data2['a'].plot.line( ls = '-.', c='red' ,figsize=(12,5),lw=2,title= 'My Plot'))   #single line
# print(data2.plot.line(figsize=(12,3),lw=5))         # multiple lins

"""Area plot"""
# print(data2.plot.area())
# print(data2.plot.area(alpha=0.4))
# print(data2.plot.area(stacked=False,alpha=0.4))

"""Scater plot"""
# print(data1.plot.scatter(x='A',y='B'))
# print(data1.plot.scatter(x='A',y='B',alpha=0.4))
# print(data1.plot.scatter(x='A',y='B',c='C',cmap='coolwarm'))   # much more


"""Box Plot"""
# print(data2.plot.box())
# print(data2.boxplot())
# print(data2[['a','b']].boxplot(grid=False))

"""Kernal Destination Estimation"""
# data2.plot.kde()
# data1.plot.kde()

plt.show()