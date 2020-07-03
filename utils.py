
'''
Utilities functions for the framework.
'''
import pandas as pd
import numpy as np
import os
import argparse
import torch
import math
import numpy.linalg as linalg
import matplotlib.pyplot as plt

import pickle
import warnings
import sklearn.metrics
warnings.filterwarnings('ignore')

import pdb

#torch.set_default_tensor_type('torch.DoubleTensor')
#torch_dtype = torch.float64 #torch.float32

res_dir = 'results'
data_dir = 'data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu' #set to CPU here if checking timing for fair timing comparison

def parse_args():
    parser = argparse.ArgumentParser()
    #    
    parser.add_argument("--verbose", dest='verbose', action='store_const', default=False, const=True, help='Print out verbose info during optimization')
    parser.add_argument("--seed", dest='fix_seed', action='store_const', default=False, const=True, help='Fix seed for reproducibility and fair comparison.')
    parser.add_argument("--exp_wt", dest='use_exp_wt', action='store_const', default=False, const=True, help='Fix seed for reproducibility and fair comparison.')
    parser.add_argument("--do_svd", action='store_const', default=False, const=True, help='use svd')
    parser.add_argument("--method", type=str, default='hypergcn', help='which baseline method')
    parser.add_argument("--kfold", default=3, type=int, help='for k-fold cross validation')
    parser.add_argument("--predict_edge", action='store_const', default=False, const=True, help='whether to predict edges')
    parser.add_argument("--edge_linear", action='store_const', default=False, const=True, help='linerity')
    parser.add_argument("--alpha_e", default=0, type=float, help='alpha')
    parser.add_argument("--alpha_v", default=0, type=float, help='alpha')
    parser.add_argument("--dropout_p", default=0.3, type=float, help='dropout')
    parser.add_argument("--n_layers", default=1, type=int, help='number of layers')
    parser.add_argument("--dataset_name", type=str, default='cora', help='dataset name')
    
    
    
    opt = parser.parse_args()
    if opt.predict_edge and opt.dataset_name != 'citeseer':
        raise Exception('edge prediction not currently supported on {}'.format(opt.dataset_name))
    return opt

def readlines(path):
    with open(path, 'r') as f:
        return f.readlines()
    

def get_label_percent(dataset_name):
    if dataset_name == 'cora':
        return .052
    elif dataset_name == 'citeseer':
        return .15 
    elif dataset_name == 'dblp':
        return .04
    else:
        raise Exception('dataset not supported')
    
    
def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
    
