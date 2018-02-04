import numpy as np
import csv
from matplotlib import pyplot as plt
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier


#======= Load Data ========#
def read_dataset(input_csv_file, isTrain=True):
    '''

    :param input_csv_file:
    :param isTrain: whether the input is a training set
    :return: a
    '''
    print('Loading data...')
    X = []
    y = []
    offset = 0
    if not isTrain:
        offset = 1

    with open(input_csv_file) as csv_file:
        csvReader = csv.reader(csv_file)
        next(csvReader)
        for row in csvReader:
            row_np = np.array(row)
            X.append(row_np[1 - offset:])
            y.append(row_np[0])
    X = np.array(X).astype(int)
    y = np.array(y).astype(int)
    return X, y


def bounding_box(img, width=28, height=28):
    img = img.reshape(height, width)
    a = np.where(img != 0)
    left, right, top, bottom = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return img[left:right+1, top:bottom+1]


def resize_img(img, w_target=20, h_target=20):
    h_origin, w_origin = img.shape
    new_img = np.zeros((h_target, w_target))
    w_ratio = float(w_origin) / w_target
    h_ratio = float(h_origin) / h_target
    for i in range(h_target):
        for j in range(w_target):
            ph = int(math.floor(i * h_ratio))
            pw = int(math.floor(j * w_ratio))
            new_img[i, j] = img[ph, pw]
    return new_img.astype(int).reshape(w_target*h_target)


def stretched_bd_box(dataset):
    print('Preprocessing data...')
    return np.array([resize_img(bounding_box(img_vec)) for img_vec in dataset])


def train_NB(feat, label, proc=True, dist='Gaussian'):
    if dist == 'Gaussian':
        clf = GaussianNB()
    elif dist == 'Bernoulli':
        clf = BernoulliNB()
    else:
        print('No such distribution')
        return
    if proc:
        feat = stretched_bd_box(feat)
    print('Training NB...')
    clf.fit(feat, label)
    return clf

def train_RF(feat, label, trees, depth, proc=True):
    clf = RandomForestClassifier(n_estimators=trees, max_depth=depth)
    if proc:
        feat = stretched_bd_box(feat)
    print('Training RF...')
    return clf.fit(feat, label)



def predict(model, test_data, proc=True):
    if proc:
        test_data = stretched_bd_box(test_data)

    return model.predict(test_data)


def eval(pred, label):
    a = pred == label
    return sum(a) / float(len(pred))



X, y = read_dataset("train.csv", True)
# a = stretched_bd_box(X)
# print(a.shape)
# print(X.shape)
clf = GaussianNB()

train_feat = X[:30000]
test_feat = X[30000:]
train_label = y[:30000]
test_label = y[30000:]



# model = train_NB(train_feat, train_label, True, 'Bernoulli')
model = train_RF(train_feat, train_label, 30, 16)
pred = predict(model, test_feat)
print(eval(pred, test_label))


# print(stretched_bd_box(X))
# idx = 24
# print(y[idx])
# img = X[idx]
# # plt.imshow(img.reshape((28, 28)))
# plt.imshow(resize_img(bounding_box(img)))
# plt.show()
# # bd_img = bounding_box(img)
# # print(bd_img)
# # print(bd_img.shape)
# #
# # print('==========================')
# #
# # print(resize_img(bd_img))
# # print(resize_img(bd_img).shape)



