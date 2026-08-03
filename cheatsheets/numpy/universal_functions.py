import numpy as np

arr = np.arange(12) 

np.sum(arr) # sum all elements
np.sum(arr, axis=0) # sum along rows (column-wise) → shape (cols,)
np.sum(arr, axis=1) # sum along columns (row-wise) → shape (rows,)
np.mean(arr) # average
np.std(arr) # standard deviation
np.min(arr) # minimum
np.argmax(arr) # index of maximum
np.unique(arr) # sorted unique values
np.sort(arr) # sorted copy