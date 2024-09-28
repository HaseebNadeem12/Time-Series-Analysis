"""NumPy is Used to perform all sorts of mathematical operation
   we can not perform these operation by useing list,tuples or sets

   In array number of element must be same"""


import numpy as np

# Create a NumPy array
array1 = np.array([1 ,2,3,4,5])
array2 = np.array([10,8,6,4,2])


"""give equal difference between number"""
array3 = np.linspace(0,1,4 )
# array_me = np.r
# print(array3)
array4 = np.arange(0,18,2)      #forms an array from 0 to 10 with the steo size of 2
# print(array4)

arr = np.array([array1,array2])             #creating 2 dimension array
arr12 = np.array([[1,2,3,4,5],[9,8,7,6,5]])
# print(arr)

arr1 = np.array(array1[0:4])    #slicing of an array
# print(arr1)
arr3 = np.array([[12,13,14,15,16],[11,17,18,19,10]]) # 2 dimensional array
# print(arr3)

"""Basic operation on arrays in NumPy"""
# print(array1*array2 )
# print(array1+array2 )
# print(array3)
# print(array4)
# print(arr[0:4,0:4])              # slicing 2 dimensional array
# print(arr[1,2:4])
# print(arr1)
# print(np.shape(arr3))          # tell us the shape
# print(np.size(arr3))
# print(arr3.dtype)

"""Mathematical operation and functions on Array"""
# # import numpy as np
#
# a = np.array([2,3,4,5,6])
# b = np.array([9,8,7,6,5])
# c = np.array([2])
# d = np.array([4, 9 ,16, 25, 36])

# print(a + b)
# print(a * b)
# print(a / b)
# print(np.)
# print(np.power(a,b))
# print(np.sqrt(d))


"""Combining and spliting an Array"""
# import numpy as np

# a = np.array([[2,3,4],[5,6,7]])
# b = np.array([[9,8,7],[6,5,8]])
# c = np.array([1,2,3,4,5,6,7,8])

# print(np.concatenate([a,b]))
# print(np.concatenate([a,b],axis=1))   #row wise concatinate
# print(np.concatenate([a,b],axis=0))   #coloumn wise concatinate

# ans = np.split(c,4) # splitting of an array into 4 smaller arrays
# print(ans[0])
# # print(ans)

"""Adding and Removing element in an array(use append, insert, pop, remove)"""
# # import numpy as np
#
# a = np.array([[2,3,4],[5,6,7]])
# print(a)
# b = np.array([[9,8,7],[6,5,8]])
# c = np.array([12,13,14,15,16,17,18,19])
#
# # print(np.append(c,[2,31,4221]))          # append
# # print(np.insert(a,2, [59,98,54] ))   # Insert
# print(np.insert(a,2, [59], axis=1 ))
# print(np.delete(c,1))                       # Delete

"""Sort, seach and filter of an array"""
# # import numpy as np
#
# a = np.array([1,2,11,9,2,7,4,0])
# b = np.array([7,4,9,3,1,7,3,5])
#
# # print(np.sort(a))     # sorting of an array (perminant change)
# # no_change = sorted(a) # not effect the original array
# # print(no_change)
# print(np.where(a == 2))   # Tell us the possition value of the element in array
# res = np.where(a%2 == 0)  # find which element is divisible by 2
# print(res)


"""Agregating Function in NumPy"""
# # import numpy as np
#
# a = np.array([1,2,11,9,2,7,4,0])
# b = np.array([7,4,9,3,1,7,3,5])
#
# print(np.sum(a))
# print(np.max(a))
# print(np.size(b))
# # print(np.cumsum(b))  #cumilative sum
# # print(np.cumprod(a)) # cumilative product
# print(np.mean(a))
#
# ans = a*b
# print(np.sum(ans))

"""Statistical function in array"""
# import numpy as np
# import statistics as stats
#
# a = np.array([1,2,3,4,5,6,7])
# b = np.array([1,4,9,16,25,36,49])
#
# print(np.mean(a))
# print(np.median(a))   # after sorting the array then it will give central value
# print(stats.mode(a))  # most accurance value
# print(np.std(a))      # standard deviation(mean value se actual value kitni door hen )
# print(np.var(a))      # variance (square of standard deviation)
#
# print(np.corrcoef([a,b]))    #corelation between two values