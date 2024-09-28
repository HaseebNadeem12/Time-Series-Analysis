"""Data_Frame"""
import pandas as pd
import numpy as np

# Creating list, array, and dictionary
my_list = [20,40,60,80]
arr = np.array([1,3,5,6])
my_dic = { 'a':10,
           'b':20,
           'c':30,
           'd':40,}

# ser1 = pd.Series(data=my_list,index=['USA','Germany','Japan','China'])
#
# ser2 = pd.Series(data=my_list,index=['USA','Itly','Japan','China'])

# print(ser1)
# print(ser2)
# print(ser1+ser2)

"""DATA_FRAME"""
fix = np.random.seed(101)
my_data_frame = pd.DataFrame(np.random.randn(4,4),index='A B C D'.split(),columns='W X Y Z'.split())
print(my_data_frame)
#
# # print(my_data_frame['Y'])
# my_data_frame['New'] = my_data_frame['X']+my_data_frame['Z']
# print(my_data_frame)
# print(my_data_frame.drop('Z',axis=1,inplace=True))
# print(my_data_frame)
# print(my_data_frame.drop('C',axis=0))

"""SLECTING COLOUMN"""
# print(my_data_frame[['X','Y']])

"""SLECTING ROWS"""
# print(my_data_frame.loc[['A','B']])

"""Slecting subset of Rows and coloumn"""
# print(my_data_frame.loc['A','Y'])
# print(my_data_frame.loc[['A','B'],['X','Y']])

"""Conditional Statement"""
# # print(my_data_frame)
#
# # my_data_frame > 0.5
# print(my_data_frame[my_data_frame > 0.5] )
# print('-'*100)
#
# print(my_data_frame[my_data_frame['X'] > 0.5])
# print('-'*100)
#
# print( my_data_frame [my_data_frame ['X'] > 0.5 ] [['Y','X']] )
# print('-'*100)
#
# print(my_data_frame[(my_data_frame['W'] > 0.5 ) & (my_data_frame['X'] > 0.5 )][['Y','Z']])

"""More Index Detail"""
# print(my_data_frame)

# Setting new index
# my_data_frame['New_Index'] = 'AB CD EF GH'.split()
# print(my_data_frame.set_index('New_Index'))
# print('-'*100)
# # Reset the new index
# print(my_data_frame.reset_index())
# print('-'*100)

"""Data Frame Summary"""
# print(my_data_frame.info)
# print(my_data_frame.describe())