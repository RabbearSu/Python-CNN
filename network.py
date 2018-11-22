'''
Description: Build the entire network

Author: Yange Cao
Version: 1.0
Date: Novemberer 15th, 2018
'''
from forward import *
from backward import *
from utils import *

def network(inputs, targets, parameters, conv_s, pool_s, pool_f):

    (W1, W2, W3, W4, b1, b2, b3, b4) = parameters

    ############### Forward Operation ################
    Z1, cache_conv1 = conv_forward(A_prev=inputs, W=W1, b=b1, conv_s=conv_s)  # 1st convolution layer forward pass
    A1 = relu_forward(Z1)  # 1st convolution layer activation 
    Z2, cache_conv2 = conv_forward(A_prev=A1, W=W2, b=b2, conv_s=conv_s)  # 2nd convolution layer forward pass
    A2 = relu_forward(Z2)  # 2nd convolution layer activation 
    pool, cache_pool = maxpool_forward(A_prev=A2, pool_f=pool_f, pool_s=pool_s)  # Pooling layer forward pass

    FL = pool.reshape(pool.shape[0], -1).T  # flatten 'pool' layer and transpose to (9216,m) 
    probs, cache_fc = fc_forward(X=FL, paras_fc=(W3, b3, W4, b4))  # forward pass in fully connected layers

    ################## Compute Cost ##################

    m = targets.shape[-1] # number of samples
    loss = (1/m) * categoricalCrossEntropy(targets, probs)

    ################ Backward Operation ##############
    (dW3, db3, dW4, db4), dFL = fc_back(AL=probs, X=FL, Y=targets, cache=cache_fc, paras=(W3, b3, W4, b4))  # backward pass in fully connected layers

    dpool = dFL.T.reshape(pool.shape)  # restore the shape of dpool

    dA2 = maxpool_back(dA=dpool, cache=cache_pool)  
    dZ2 = dA2 * relu_back(Z2)  

    dA1, dW2, db2 = conv_back(dZ=dZ2, cache=cache_conv2)  
    dZ1 = dA1 * relu_back(Z1)  

    d_inputs, dW1, db1 = conv_back(dZ=dZ1, cache=cache_conv1)  

    grads = (dW1/m, dW2/m, dW3/m, dW4/m, db1/m, db2/m, db3/m, db4/m)
    return grads, loss


def adam(img_dim, img_depth, batch, paras, lr, beta1, beta2, cost):

    (W1, W2, W3, W4, b1, b2, b3, b4) = paras

    #print(batch[:,0:-1].shape)
    X = batch[:, 0:-1].reshape(len(batch), img_dim, img_dim, img_depth)
    # one-hot encode
    Y = batch[:, -1].astype(int)
    Y = np.eye(10)[Y].T
    # get grads and loss 
    grads, loss = network(X, Y, paras, conv_s=1, pool_f=2, pool_s=2)
    (dW1, dW2, dW3, dW4, db1, db2, db3, db4) = grads
    
    # do adam optimisation
    v1 = np.zeros(W1.shape)
    v2 = np.zeros(W2.shape)
    v3 = np.zeros(W3.shape)
    v4 = np.zeros(W4.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)

    s1 = np.zeros(W1.shape)
    s2 = np.zeros(W2.shape)
    s3 = np.zeros(W3.shape)
    s4 = np.zeros(W4.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)

    v1 = beta1 * v1 + (1 - beta1) * dW1  # momentum update
    s1 = beta2 * s1 + (1 - beta2) * dW1 ** 2  # RMSProp update
    W1 -= lr * v1 / np.sqrt(s1 + 1e-7)  # combine momentum and RMSProp to perform update with Adam

    bv1 = beta1 * bv1 + (1 - beta1) * db1
    bs1 = beta2 * bs1 + (1 - beta2) * db1  ** 2
    b1 -= lr * bv1 / np.sqrt(bs1 + 1e-7)

    v2 = beta1 * v2 + (1 - beta1) * dW2
    s2 = beta2 * s2 + (1 - beta2) * dW2 ** 2
    W2 -= lr * v2 / np.sqrt(s2 + 1e-7)

    bv2 = beta1 * bv2 + (1 - beta1) * db2
    bs2 = beta2 * bs2 + (1 - beta2) * db2 ** 2
    b2 -= lr * bv2 / np.sqrt(bs2 + 1e-7)

    v3 = beta1 * v3 + (1 - beta1) * dW3
    s3 = beta2 * s3 + (1 - beta2) * dW3  ** 2
    W3 -= lr * v3 / np.sqrt(s3 + 1e-7)

    bv3 = beta1 * bv3 + (1 - beta1) * db3
    bs3 = beta2 * bs3 + (1 - beta2) * db3  ** 2
    b3 -= lr * bv3 / np.sqrt(bs3 + 1e-7)

    v4 = beta1 * v4 + (1 - beta1) * dW4
    s4 = beta2 * s4 + (1 - beta2) * dW4 ** 2
    W4 -= lr * v4 / np.sqrt(s4 + 1e-7)

    bv4 = beta1 * bv4 + (1 - beta1) * db4
    bs4 = beta2 * bs4 + (1 - beta2) * db4 ** 2
    b4 -= lr * bv4 / np.sqrt(bs4 + 1e-7)

    paras_updated = (W1, W2, W3, W4, b1, b2, b3, b4)
    cost.append(loss)
    return paras_updated, cost
