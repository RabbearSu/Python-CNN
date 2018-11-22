'''
Description: the train function for the  CNN

Author: Yange Cao
Version: 1.0
Date: Novemberer 15th, 2018
'''

from network import *
from tqdm import tqdm
import pickle

def train(img_dim,
          img_depth,
          f1,
          f2,
          num_f1,
          num_f2,
          batch_size,
          lr,
          beta1,
          beta2,
          num_epochs,
          save_path):

    paras = init_paras(img_dim, img_depth, f1, f2, num_f1, num_f2)
    train_data, test_data = get_data_cifar10()
    cost = []
    print("LR:" + str(lr) + ", Batch Size:" + str(batch_size))

    for epoch in range(num_epochs):
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for x, batch in enumerate(t):
            params, cost = adam(img_dim, img_depth, batch, paras, lr, beta1, beta2, cost)
            t.set_description("Cost: {}".format(cost[-1]))

    to_save = [params, cost]
    with open(save_path, 'wb') as file:
        pickle.dump(to_save, file)

    return cost






