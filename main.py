'''
Description: main function for the  CNN

Author: Yange Cao
Version: 1.0
Date: Novemberer 15th, 2018
'''

from network import *
from train import *
#from utils import *
import numpy as np

cost = train(img_dim=32,
             img_depth=3,
             f1=5,
             f2=5,
             num_f1=8,
             num_f2=8,
             batch_size=32,
             lr=0.01,
             beta1=0.95,
             beta2=0.99,
             num_epochs=2,
             save_path='cifar10.pkl')
