import numpy as np # For generating data
from sklearn.linear_model import LinearRegression # For creating and training the model
from sklearn.metrics import mean_absolute_error # For making predictions and measuring error
from sklearn.model_selection import train_test_split # For training

model = LinearRegression()

# features = column
# samples = row

house_sizes = np.random.uniform(30.0, 200.0, 50).reshape(-1, 1) 
# np.random.uniform() - low (inclusive), high (exclusive), count
# .reshape(-1, 1) - scikit‑learn requires features to be 2D
# -1 to automatically adjust based on how many values the array contains
noise = np.random.normal(0, 20000, 50) # mean, std, count
price = house_sizes * 1500 + 20000 + noise



