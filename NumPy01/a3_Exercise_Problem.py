import numpy as np

"""create an array of 10 zeros"""
# array1 = np.zeros(10)
# print(array1)

"""create an array of 10 ones"""
# array1 = np.ones(10)
# print(array1)

"""create an array of 10 fives"""
# array1 = np.ones(10)*5
# print(array1)

"""create an array of integer between 10 and 50"""
# array1 = np.arange(10,51)
# print(array1)

"""create an array of all even integers between 10 and 50"""
# array1 = np.arange(10,51,2)       #changing the step size
# print(array1)

"""create a matrix of 3 by 3"""
# array1 = np.arange(0,9).reshape(3,3)
# print(array1)

"""create an identity matrix of 3 by 3"""
# array1 = np.eye(3)
# print(array1)

"""generate 1 random number between 0 and 1"""
# array1 = np.random.rand(1)
# print(array1)

"""create an array of 25 random number between 0 and 1"""
# array1 = np.random.randn(25)
# print(array1)

"""create an identity matrix of 3 by 3"""
# array1 = (np.arange(1,101)/100).reshape(10,10)
# print(array1)

"""Create an array of linearly spaced between 0 & 1"""
# array1 = np.linspace(0,1,20)
# print(array1)

"""Create a matrix and take out the slice"""
# array1 = np.arange(1,26).reshape(5,5)
#
# print(array1[2:5])
# print(array1[3,4])
#
# print(array1.sum())
# print(array1.std())
# print(array1.sum(axis=0))

"""Fixed random number"""
# array1 = np.random.seed(104)
# array1 = np.random.rand(1)
# print(array1)