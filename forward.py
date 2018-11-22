'''
Description: forward operations for a CNN

Author: Yange Cao
Version: 1.0
Date: Novemberer 15th, 2018
'''
import numpy as np
from utils import *

def conv_forward(A_prev, W, b, conv_s):
    #print('W.shape:{}, b.shape:{}'.format(W.shape,b.shape))
    # abstract parameters
    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
    (f, f, n_c_prev, n_c) = W.shape
    # the output shape 
    n_h = int((n_h_prev - f) / conv_s) + 1
    n_w = int((n_w_prev - f) / conv_s) + 1
    # initialize output using zeros
    z = np.zeros((m, n_h, n_w, n_c))
    # go through all demensions of z
    for i in range(m):
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    h_start = h * conv_s
                    h_end = h_start + f
                    w_start = w * conv_s
                    w_end = w_start + f
                    a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, :]
                    z[i, h, w, c] = np.sum(a_prev_slice * W[..., c]) + b[:,c]
    assert (z.shape == (m, n_h, n_w, n_c))
    cache = (A_prev, W, b, conv_s)
    return z, cache

def maxpool_forward(A_prev, pool_f, pool_s):
    # abstract parameters
    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
    # the shape after pooling
    n_h = int((n_h_prev - pool_f) / pool_s) + 1
    n_w = int((n_w_prev - pool_f) / pool_s) + 1
    n_c = n_c_prev
    # initial output using zeros
    z = np.zeros((m, n_h, n_w, n_c))
    # go through all demensions of z
    for i in range(m):
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    h_start = h * pool_s
                    h_end = h_start + pool_f
                    w_start = w * pool_s
                    w_end = w_start + pool_f
                    a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                    z[i, h, w, c] = np.max(a_prev_slice)
    assert (z.shape == (m, n_h, n_w, n_c))
    cache = (A_prev, pool_f, pool_s)
    return z, cache

# forward pass in fully connected layers
def fc_forward(X, paras_fc):
    (W3, b3, W4, b4) = paras_fc
    Z3 = np.dot(W3, X) + b3
    A3 = relu_forward(Z3)
    Z4 = np.dot(W4, A3) + b4
    A4 = softmax(Z4)
    cache = (A4, Z4, A3, Z3)
    return A4, cache

# softmax entropy
def categoricalCrossEntropy(labels, probs):
    return -np.sum(labels * np.log(probs))
