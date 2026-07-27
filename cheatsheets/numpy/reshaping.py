import numpy as np

arr = np.arange(12)     # [0, 1, ..., 11]

arr.reshape(3, 4)       # 3×4 - need to preserve total elements
arr.reshape(-1, 4)      # 3×4 - -1 = "figure this dimension out"
arr.ravel()             # flatten to 1D - returns view when possible
arr.flatten()           # flatten to 1D - always returns a copy

# Transpose
b = arr.reshape(3, 4)
b.T                     # 4×3 - swap axes

# Stacking
np.vstack([b, b])       # vertical stack (row-wise)
np.hstack([b, b])       # horizontal stack (column-wise)