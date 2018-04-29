import numpy as np


x = np.array([1, 3, 2, 4, 2])
y = np.array([2, 4, 3, 1, 3])

a = np.arange(0, 36).reshape((6, 6))
# print(a)
# print(a[x, y])

# for i, j in zip(x, y):
#     print(i)
#     print(j)
#     print('===')

# for dx in range(-1, 2, 1):
#     for dy in range(-1, 2, 1):
#         print("{0}:{1}".format(dx, dy))


def neighbors(row, col):
    '''
    :param row: the row of a pixel
    :param col: the col of a pixel
    :return: a 2d matrix with shape (n, 2)
             where n is number of neighbors of a pixel
             and each row is the row and col of its neighbors
    '''
    neighbors = []
    for d_row in range(-1, 2, 1):
        for d_col in range(-1, 2, 1):
            n_row = row + d_row
            n_col = col + d_col
            if (d_row | d_col) != 0 and n_row >= 0 and n_row < 6 and n_col >= 0 and n_col < 6:
                neighbors.append([n_row, n_col])
    return np.array(neighbors)

# print(neighbors(1,1))
print(a)
nei = neighbors(0, 1)
# print(nei)

print(a[nei[:, 0], nei[:, 1]])

