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
        self.energy = np.zeros((20, 11))
        self.energy_samples = None

        self.load_data()
        self.preprocessing()
        self.add_noise()

    def preprocessing(self):
        '''
        :return: return write-black photo
        '''
        print("Preprocessing...")
        idx = self.train_images < 128
        self.train_images = np.ones(self.train_images.shape)
        self.train_images[idx] = -1
        self.noise_images = np.array(self.train_images)

    def add_noise(self):
        print("Adding noise...")
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
        print("Loading data...")
        # Load Update Order Coordinate
        nclos = 784
        order_coo = pd.read_csv(self.data_dir + '/UpdateOrderCoordinates.csv', sep=',', usecols=range(1, nclos + 1)).as_matrix()
        order_coo.astype(np.uint8)
        self.update_order_coo = order_coo
        # print(order_coo[0, -2:])

        # Load Inital Parameters Model (initial Q)
        self.Q_init = pd.read_csv(self.data_dir + '/InitialParametersModel.csv', sep=',', dtype=float, header=None).as_matrix()
        # print(Q[0][0])

        self.energy_samples = pd.read_csv(self.data_dir + '/EnergySamples.csv', sep=',', dtype=float, header=None).as_matrix()

    def update(self, iters=10):
        for i in range(0, self.nimgs * 2 - 1, 2):
            Q = np.array(self.Q_init)
            img_idx = i // 2
            self.energy[img_idx, 0] = self.variational_free_energy(Q, img_idx)
            for iter in range(1, iters + 1):
                self.update_one_img(Q, img_idx)
                self.energy[img_idx, iter] = self.variational_free_energy(Q, img_idx)

    def update_one_img(self, Q, img_idx):
        coo_row = img_idx * 2
        coo_col = img_idx * 2 + 1
        pow = 0
        for row, col in zip(self.update_order_coo[coo_row], self.update_order_coo[coo_col]):
            # Hidden Neighbors
            neighbor_idx = self.neighbors(row, col) # a matrix with shape (n, 2)
            pow += np.sum(self.theta_HH * (2 * Q[neighbor_idx[:, 0], neighbor_idx[:, 1]] - 1))
            # Observed Neighbors
            pow += self.theta_HX * self.noise_images[img_idx][row, col]
            temp = np.exp(pow)
            pi = temp / (temp + 1 / temp)
            Q[row][col] = pi

    def variational_free_energy(self, Q, img_idx):
        eps = 10**-10
        term1 = np.sum(Q * np.log(Q + eps)) + np.sum((1 - Q) * np.log((1 - Q) + eps))
        # Hidden Neighbors
        E_q = 2 * Q - 1 # E_q Energy
        hidden = 0
        for row in range(self.img_shape[0]):
            for col in range(self.img_shape[1]):
                neighbor_idx = self.neighbors(row, col)
                hidden += self.theta_HH * E_q[row, col] * np.sum(E_q[neighbor_idx[:, 0], neighbor_idx[:, 1]])
        observed = np.sum(self.theta_HX * E_q * self.noise_images[img_idx])
        term2 = hidden + observed
        return term1


    def neighbors(self, row, col):
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
                if (d_row | d_col) != 0 and n_row >= 0 and n_row < self.img_shape[0] and n_col >= 0 and n_col < self.img_shape[1]:
                    neighbors.append([n_row, n_col])
        return np.array(neighbors)










def main():
    data_dir = "SupplementaryAndSampleData"

    BM = ImageDenoising(data_dir)
    # BM.update(iters=10)
    # print(BM.energy)
    # res = BM.test_add_noise()
    # print(res[1:])

    print(BM.variational_free_energy(BM.Q_init, 0))


    # imgs = BM.train_images
    # plt.imshow(res[0], cmap='gray')
    # plt.show()
    # print(imgs[0])
    # scipy.misc.toimage(scipy.misc.imresize(imgs[0, :, :] * -1 + 256, 10.))


if __name__ == "__main__":
    main()
