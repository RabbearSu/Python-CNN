'''
Description: Backward operations for a CNN

Author: Yange Cao
Version: 1.0
Date: Novemberer 15th, 2018
'''

import numpy as np
from utils import *

def conv_back(dZ, cache):
    # Abstract parameters from cache
    (A_prev, W, b, conv_s) = cache
    (f, f, n_c_prev, n_c) = W.shape
    (m, n_h, n_w, n_c) = dZ.shape
    # initialize the outputs
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros((1, n_c))
    for i in range(m):
        for c in range(n_c):
            for h in range(n_h):
                for w in range(n_w):
                    h_start = h * conv_s
                    h_end = h_start + f
                    w_start = w * conv_s
                    w_end = w_start + f
                    a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, :]
                    # get the derivatives
                    dW[:, :, :, c] += a_prev_slice * dZ[i, h, w, c]
                    db[:, c] += dZ[i, h, w, c]
                    dA_prev[i, h_start:h_end, w_start:w_end, :] += W[:, :, :, c] * dZ[i, h, w, c]

    assert (dA_prev.shape == A_prev.shape)
    return dA_prev, dW, db

def maxpool_back(dA, cache):
    # abstract cache
    (A_prev, pool_f, pool_s) = cache
    (m, n_h, n_w, n_c) = dA.shape
    # initialize output
    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):
        for c in range(n_c):
            for h in range(n_h):
                for w in range(n_w):
                    h_start = h * pool_s
                    h_end = h_start + pool_f
                    w_start = w * pool_s
                    w_end = w_start + pool_f
                    a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                    # Create the mask from a_prev_slice
                    mask = create_mask(a_prev_slice)
                    dA_prev[i, h_start:h_end, w_start:w_end, c] += mask * dA[i, h, w, c]
    assert (dA_prev.shape == A_prev.shape)
    return dA_prev

def fc_back(AL, cache, X, Y, paras):
    (A4, Z4, A3, Z3) = cache
    (W3, b3, W4, b4) = paras

    dZ4 = AL - Y  # (10,m) m is number of samples
    dW4 = np.dot(dZ4, A3.T)  # (10,128)
    db4 = np.sum(dZ4, axis=1).reshape(b4.shape)  # (10,1)

    dA3 = np.dot(W4.T, dZ4)  # (128,m)
    dZ3 = dA3 * relu_back(Z3)  # (128,m)
    dW3 = np.dot(dZ3, X.T)  # (128,9216)
    db3 = np.sum(dZ3, axis=1).reshape(b3.shape)  # (128,1)

    dX = np.dot(W3.T, dZ3)

    d_paras = (dW3, db3, dW4, db4)
    return d_paras, dX
