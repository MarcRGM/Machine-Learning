import numpy as np

a = np.array([1, 2, 3]) # 1D array (vector)
b = np.array([[1, 2], [3, 4]]) # 2D array (matrix)

# Convenience constructors
np.zeros((3, 4)) # 3×4 matrix of zeros 
np.ones((2, 3)) # 2×3 matrix of ones 
np.eye(3) # 3×3 identity matrix
np.arange(0, 10, 2) # (Start: Inclusive, End: Exclusive, Steps) = [0, 2, 4, 6, 8]
np.linspace(0, 1, 5) # (Start: Inclusive, End: Inclusive, N) = [0., 0.25, 0.5, 0.75, 1.] - N evenly-spaced points

# Random arrays
np.random.seed(42) # locks the random sequence; 42 is just a placeholder value
np.random.rand(3, 2) # Uniform distribution [0.0, 1.0) - Always positive, always less than 1
np.random.randn(3, 2) # Normal distribution (Mean = 0, Std Dev = 1) - Can be negative, zero, or greater than 1
np.random.randint(0, 10, size=(3, 2)) # integers in [0, 10)

# [] Closed interval - Both the start and end numbers are included in the range.
# [) Half-open / Half-closed interval - The start number is included, but the end number is excluded.
