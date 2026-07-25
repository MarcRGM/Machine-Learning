import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape # (2, 3) - dimensions
arr.ndim # 2 - number of dimensions
arr.size # 6 - total elements
arr.dtype # dtype('int64') - data type of elements

# Indexing - same as Python lists but with multi-axis support
arr[0, 1] # 2 - row 0, column 1
arr[:, 1] # array([2, 5]) - all rows, column 1 (:) = "entire axis"
arr[0, :] # array([1, 2, 3]) - row 0, all columns

# Boolean indexing - good for filtering
arr[arr > 3] # array([4, 5, 6]) - all elements > 3
arr[arr % 2 == 0]  # array([2, 4, 6]) - all even elements

# Indexing with integer arrays
arr[[0, 1], [2, 0]] # [rows], [cols] = array([3, 4]) - elements at (0,2) and (1,0) 
