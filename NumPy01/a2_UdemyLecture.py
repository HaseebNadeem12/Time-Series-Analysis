"""Udemy Lecture"""
from array import array

import numpy as np

array1 = [[1,2,3],[4,5,6],[7,8,9]]  # Two dimensional Array
My_matrix = np.array(array1)

# print(np.arange(0,10,3))   #create array of given range
# print(My_matrix.sum())   #can perform tons of operation

# print(np.ones((4,10)))       #create matrix
# print(np.ones((4,10)) + 5)   # can perform multiple operation
# print(np.ones(4)*3)
#
# print(np.linspace(0,10,3)) # create an array with evenly spaces
# print(np.eye(5))    #give 5 by 5 identity matrix

"""Random number"""
# print(np.random.rand(4))     #give 4 equal number between 0 & 1
# print(np.random.rand(5,5))   #give random number between 0 & 1 of 5 by 5 matrix
# print(np.random.randn(10))   #give 10 random number between 0 & 1(search)
#
# np.random.seed(42)           #Random number will not be changed with SEED
# print(np.random.randint(0,100,5))
# #(StartValue,EndValue,HowManyNoYouWant) Generate random number
# #
# arr = np.arange(25)
# print(arr)
# print(arr.reshape(5,5))   # Converting 1D array into 2D array(5 by 5)
# #
# print(np.argmin(arr))     #return possition of max and min value


"""Numpy indexcing and slection"""
# # import numpy as np
#
# array1 = np.arange(0,11)
# print(array1[2:6])
#
# array2 = np.array([[1,2,3,4],[8,7,6,5],[3,5,7,9]])
# print(array2[1,3])      #Access to one element [row number, element number]
# print(array2[1,1:3])    #access to 2 or 3 element


"""Conditional slection of an array"""

arr = np.arange(0,9)
arr1 = arr.reshape(3,3)
bool_arr = arr>4           #access boolean values according to condition
# print(bool_arr)
# print(arr[bool_arr])       #acces numeric values
#
# print(arr[arr>4])               #shorter method
# print(arr[arr<6])
#
# print(array.sum(arr,axis=0))      #give sum accross colounms
print(arr1.sum(axis=1))            #give sum accross rows




