import numpy as np

# convert data type into float to convert missing datas into nan value (np.nan)

arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan])

# Standard versions - NaN poisons the result
np.mean(arr) # nan - useless with missing data
np.sum(arr) # nan
np.std(arr) # nan

# NaN-aware versions - ignore NaN values
np.nanmean(arr) # 2.333... - mean of [1, 2, 4] = 7/3
np.nansum(arr) # 7.0 - sum of non-NaN values
np.nanstd(arr) # 1.247... - standard deviation ignoring NaN
np.nanvar(arr) # 1.555... - variance ignoring NaN
np.nanmin(arr) # 1.0 - minimum of valid values
np.nanmax(arr) # 4.0 - maximum of valid values
np.nanprod(arr) # 8.0 - product (1*2*4)
np.nanmedian(arr) # 2.0 - median of [1, 2, 4]
np.nanquantile(arr, 0.5) # 2.0 - 50th percentile (same as median)
np.nanpercentile(arr, 50) # 2.0 - percentile by 0-100 scale

# Works with axis= just like the standard versions
matrix = np.array([[1, 2, np.nan], [4, np.nan, 6]])
np.nanmean(matrix, axis=0) # [2.5, 2.0, 6.0] - column means
np.nanmean(matrix, axis=1) # [1.5, 5.0] - row means