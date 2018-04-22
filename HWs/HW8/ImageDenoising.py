import mnist
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd


class ImageDenoising:

    def __init__(self, dir):
        self.nimgs = 1
        self.data_dir = dir
        self.train_images = mnist.train_images()[:self.nimgs, :, :]

    def preprocessing(self):
        idx = self.train_images < 128
        self.train_images = np.ones(self.train_images.shape)
        self.train_images[idx] = -1

    def add_noise(self):
        nclos = 15
        flip_coo = pd.read_csv(self.data_dir + '/NoiseCoordinates.csv', sep=',', usecols=range(1, nclos + 1)).as_matrix()
        flip_coo.astype(np.uint8)
        for i in range(0, self.nimgs * 2 - 1, 2):
            img = self.train_images[i // 2]
            # print(img[flip_coo[i], flip_coo[i + 1]])
            # print(flip_coo)
            # print(flip_coo[i])
            # print(flip_coo[i + 1])
            img[flip_coo[i], flip_coo[i + 1]] *= -1

    def test_add_noise(self):
        org_imgs = np.array(self.train_images)
        self.add_noise()
        diff = org_imgs - self.train_images
        return diff






def main():
    data_dir = "SupplementaryAndSampleData"

    BM = ImageDenoising(data_dir)
    BM.preprocessing()
    BM.add_noise()
    res = BM.test_add_noise()
    # print(res[1:])


    imgs = BM.train_images
    plt.imshow(res[0], cmap='gray')
    plt.show()
    # print(imgs[0])
    # scipy.misc.toimage(scipy.misc.imresize(imgs[0, :, :] * -1 + 256, 10.))

if __name__ == "__main__":
    main()
