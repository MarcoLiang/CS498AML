import numpy as np
from PIL import Image
from skimage.transform import resize

img = np.zeros((6,6))
img[2, 2] = 1
img[4, 4] = 1
img[3, 3] = 1
img[1, 5] = 1

# print(img)


def bounding_box(img):
    a = np.where(img != 0)
    left, right, top, bottom = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return img[left:right+1, top:bottom+1]

# print(bounding_box(img))

def resize_img(img):
    print(type(img))

print(resize(img, (3,3)))

