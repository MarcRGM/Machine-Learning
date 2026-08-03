import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise - no loops needed
a + b # array([5, 7, 9])
a * b # array([4, 10, 18]) - element-wise multiply, NOT dot product
a ** 2 # array([1, 4, 9])
np.sqrt(a) # array([1., 1.414, 1.732])

# Broadcasting - arrays of different shapes align automatically
a + 10 # array([11, 12, 13]) - scalar broadcast to every element
matrix = np.ones((3, 4))
matrix + a.reshape(3, 1) # (3,4) + (3,1) = (3,4) 