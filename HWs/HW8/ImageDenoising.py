import mnist
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd


class ImageDenoising:

    def __init__(self, dir):
        self.nimgs = 20
        self.data_dir = dir
        self.noise_images = None
        self.train_images = mnist.train_images()[:self.nimgs, :, :]
        self.update_order_coo = None
        self.img_shape = self.train_images[0].shape
        self.Q_init = None
        self.theta_HH = 0.8
        self.theta_HX = 0.2

    def preprocessing(self):
        '''
        :return: return write-black photo
        '''
        idx = self.train_images < 128
        self.train_images = np.ones(self.train_images.shape)
        self.train_images[idx] = -1
        self.noise_images = np.array(self.train_images)

    def add_noise(self):
        nclos = 15
        flip_coo = pd.read_csv(self.data_dir + '/NoiseCoordinates.csv', sep=',', usecols=range(1, nclos + 1)).as_matrix()
        flip_coo.astype(np.uint8)
        for i in range(0, self.nimgs * 2 - 1, 2):
            img = self.noise_images[i // 2]
            img[flip_coo[i], flip_coo[i + 1]] *= -1

    def test_add_noise(self):
        org_imgs = np.array(self.train_images)
        self.add_noise()
        diff = org_imgs - self.train_images
        return diff

    def load_data(self):
        # Load Update Order Coordinate
        nclos = 784
        order_coo = pd.read_csv(self.data_dir + '/UpdateOrderCoordinates.csv', sep=',', usecols=range(1, nclos + 1)).as_matrix()
        order_coo.astype(np.uint8)
        self.update_order_coo = order_coo
        # print(order_coo[0, -2:])

        # Load Inital Parameters Model (initial Q)
        self.Q_init = pd.read_csv(self.data_dir + '/InitialParametersModel.csv', sep=',', dtype=float, header=None).as_matrix()
        # print(Q[0][0])

    def mean_field_inference(self, iters=10):
        pass

    def update(self, iters=10):
        for iter in range(iters):
            for i in range(0, self.nimgs * 2 - 1, 2):
                img_idx = i // 2
                img = self.train_images[img_idx]
                Q = np.array(self.Q_init)
                for row, col in zip(self.update_order_coo[i], self.update_order_coo[i + 1]):
                    self.Q

    def update_one_img(self, Q, img_idx):
        coo_row = img_idx * 2
        coo_col = img_idx * 2 + 1
        pow = 0
        for row, col in zip(self.update_order_coo[coo_row], self.update_order_coo[coo_col]):
            # Hidden Neighbors
            for d_row in range(-1, 2, 1):
                for d_col in range(-1, 2, 1):
                    n_row = row + d_row
                    n_col = col + d_col
                    if n_row >= 0 and n_row < self.img_shape[0] and n_col >= 0 and n_col < self.img_shape[1]:
                        pow += self.theta_HH * (2 * Q[n_row, n_col] - 1)
            # Observed Neighbors
            pow += self.theta_HX * self.noise_images[img_idx][row, col]
            temp = np.exp(pow)
            pi = temp / (temp + 1 / temp)
            Q[row][col] = pi









def main():
    data_dir = "SupplementaryAndSampleData"

    BM = ImageDenoising(data_dir)
    BM.preprocessing()
    BM.add_noise()
    res = BM.test_add_noise()
    # print(res[1:])


    imgs = BM.train_images
    # plt.imshow(res[0], cmap='gray')
    # plt.show()
    # print(imgs[0])
    # scipy.misc.toimage(scipy.misc.imresize(imgs[0, :, :] * -1 + 256, 10.))

    BM.load_data()

if __name__ == "__main__":
    main()
