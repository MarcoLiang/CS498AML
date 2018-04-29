import numpy as np


x = np.array([1, 3, 2, 4, 2])
y = np.array([2, 4, 3, 1, 3])

# a = np.arange(0, 36).reshape((6, 6))
# print(a)
# print(a[x, y])

# for i, j in zip(x, y):
#     print(i)
#     print(j)
#     print('===')

for dx in range(-1, 2, 1):
    for dy in range(-1, 2, 1):
        print("{0}:{1}".format(dx, dy))