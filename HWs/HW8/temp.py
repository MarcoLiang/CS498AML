import numpy as np


x = np.array([1, 3, 2, 4, 2])
y = np.array([2, 4, 3, 1, 3])

a = np.arange(0, 36).reshape((6, 6))
print(a)
print(a[x, y])