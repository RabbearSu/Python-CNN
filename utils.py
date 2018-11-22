'''
Description: utility functions for a CNN

Author: Yange Cao
Version: 1.0
Date: Novemberer 15th, 2018
'''
import numpy as np
import pandas as pd
import pickle

# relu activation function
def relu_forward(Z):
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    return A

# derivative of the Z before relu
def relu_back(Z):
    dZ = np.ones_like(Z)
    dZ[Z<=0] = 0
    assert(dZ.shape == Z.shape)
    return dZ

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# prepare mask for max_pooling backward
def create_mask(x):
    return (x==np.max(x))

# initialize parameters
def init_paras(img_dim, img_depth, f1, f2, num_f1, num_f2):
    num_features = int(np.square((img_dim - f1 - f2 + 2) / 2) * num_f2)

    W1 = np.random.randn(f1, f1, img_depth, num_f1) * 0.01
    W2 = np.random.randn(f2, f2, num_f1, num_f2) * 0.01
    W3 = np.random.randn(128, num_features) * 0.01
    W4 = np.random.randn(10, 128) * 0.01

    b1 = np.zeros((1, num_f1)) * 0.01
    b2 = np.zeros((1, num_f2)) * 0.01
    b3 = np.zeros((128, 1)) * 0.01
    b4 = np.zeros((10, 1)) * 0.01

    paras = (W1, W2, W3, W4, b1, b2, b3, b4)
    return paras

def get_data_mnist():
    mnist_dataframe = pd.read_csv('data\\mnist_train_small.csv', sep=',', header=None)

    labels = mnist_dataframe[0].values.reshape(20000,1)
    imgs = np.array(mnist_dataframe.iloc[:, 1:].values, dtype='float')

    # normalize
    imgs -= int(np.mean(imgs))
    imgs /= int(np.std(imgs))

    # stack imgs and labels for shuffling
    data = np.hstack((imgs, labels))
    np.random.shuffle(data)

    train_data = data[:16000, :]
    test_data = data[16000:20000, :]
    return train_data, test_data

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def get_data_cifar10():
    data_1 = unpickle('data\\data_batch_1')
    data_2 = unpickle('data\\data_batch_2')
    data_3 = unpickle('data\\data_batch_3')
    data_4 = unpickle('data\\data_batch_4')
    data_5 = unpickle('data\\data_batch_5')
    data_test = unpickle('data\\test_batch')
    # abstract imgs
    X_1 = np.array(data_1[b'data'], dtype='float')
    X_2 = np.array(data_2[b'data'], dtype='float')
    X_3 = np.array(data_3[b'data'], dtype='float')
    X_4 = np.array(data_4[b'data'], dtype='float')
    X_5 = np.array(data_5[b'data'], dtype='float')
    X_train = np.vstack((X_1, X_2, X_3, X_4, X_5))
    X_test = np.array(data_test[b'data'], dtype='float')

    # Normalize
    X_train -= int(np.mean(X_train))
    X_train /= int(np.std(X_train))
    X_test -= int(np.mean(X_test))
    X_test /= int(np.std(X_test))

    # abstract labels
    Y_1 = np.array(data_1[b'labels'])
    Y_2 = np.array(data_2[b'labels'])
    Y_3 = np.array(data_3[b'labels'])
    Y_4 = np.array(data_4[b'labels'])
    Y_5 = np.array(data_5[b'labels'])
    Y_train = np.vstack((Y_1, Y_2, Y_3, Y_4, Y_5)).reshape(50000, 1)
    Y_test = np.array(data_test[b'labels']).reshape(10000,1)

    # stack imgs and labels
    train_data = np.hstack((X_train, Y_train))

    # print(train_data.shape)
    # print(X_test.shape, Y_test.shape)
    test_data = np.hstack((X_test, Y_test))
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    return train_data, test_data




