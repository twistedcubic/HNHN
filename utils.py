
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
#import networkx as nx
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
    
def view_graph(L, soft_edge=False, labels=None, name=''):
    """
    Input:
    L: laplacian. Tensor
    labels: node labels.
    """
    plt.clf()
    if isinstance(L, torch.Tensor):        
        L = L.cpu().numpy()
    L = L.copy()
    #make diagonal 0?! negate?
    np.fill_diagonal(L, 0)
    L *= -1
    #pdb.set_trace()
    g = nx.from_numpy_array(L)
    fig = plt.figure()
    plt.axis('off')
    ax = plt.gca()
    #ax.axis('off')
    #layout = nx.kamada_kawai_layout(g)
    layout = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, layout, node_size=500, alpha=0.5, cmap=plt.cm.RdYlGn, node_color='r', ax=ax)
    if labels is None:        
        nx.draw_networkx_labels(g, layout, font_color='w', font_weight='bold', font_size=15, ax=ax)
    else:
        #labels can be determined from P
        nx.draw_networkx_labels(g, layout, labels=labels, font_color='k', font_size=12, ax=ax) #13
    if soft_edge:
        #sort edges! to determine the cutoff
        #elarge = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] > 2]
        #esmall = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] <= 2 and d['weight'] > .99] #move to args!
        elarge = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] > .5] #.5
        esmall = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] <= .5 and d['weight'] > .19] #move to args! 3, 1.9 .2
        nx.draw_networkx_edges(g, layout, edgelist=elarge, width=3.5, ax=ax)
        nx.draw_networkx_edges(g, layout, edgelist=esmall, wifth=3, ax=ax)
    else:
        nx.draw_networkx_edges(g, layout, ax=ax)
    
    fig.savefig('data/view_graph_{}.jpg'.format(name))
    print('plot saved to {}'.format('data/view_graph_{}.jpg'.format(name)))
    plt.show()

def get_label_percent(dataset_name):
    if dataset_name == 'cora':
        return .052
    elif dataset_name == 'citeseer':
        return .15 
    elif dataset_name == 'dblp':
        return .04
    else:
        raise Exception('dataset not supported')
    
def plot_confusion(tgt, pred, labels=None, name=''):
    """
    Input:
    tgt, pred: target and predicted classes.
    labels: node labels.
    """
    plt.clf()
    fig = plt.figure()
    #if isinstance(L, torch.Tensor):        
    #    L = L.cpu().numpy()
    ax = plt.gca()
    mx = sklearn.metrics.confusion_matrix(tgt, pred)
    #pdb.set_trace()
    img = plt.matshow(mx)
    path = 'data/confusion_mx_{}.jpg'.format(name)
    #plt.imsave(path, img)
    ax.legend()
    name2label = {'gw_cls':'GW', 'ot_cls':'COPT', 'combine_cls':'[COPT + GW]'}
    plt.title('{} Predictions'.format(name2label[name]), fontsize=20)
    plt.savefig(path)
    print('fig saved to ', path)


def plot_search_acc():
    '''
    Plot search acc. 
    '''
    plt.clf()
    x_l = [1, 3, 5, 10, 15]
    ot_acc = [0.9721962,0.9866296,0.9941481333,0.998,0.998]#[0.9777, .988, .994, .994, .994]
    svd_acc = [0.814787,0.894257037,0.9349997,0.9774814073,0.9888888777]#[0.8277, .905, .955, .988, 1]
    ot_std = [0.005516745247,0.003174584773,0.0002565744596,0.003464101615,0.003464101615] #[0.9777, .988, .994, .994, .994]
    svd_std = [0.01152346983,0.02409426815,0.02220130725,0.009109368462,0.009622514205] #[0.8277, .905, .955, .988, 1]

    fig = plt.figure()
    #ax = ax
    #plt.plot(x_l, ot_acc, '-o', label='COPT ')
    plt.errorbar(x_l, ot_acc, yerr=ot_std, marker='o', label='COPT sketches')
    #plt.plot(x_l, svd_acc, '-*', label='SVD  ')
    plt.errorbar(x_l, svd_acc, yerr=svd_std, marker='+', label='Spectral projections')
    
    plt.title('Classification acc of [COPT, GW] vs [spectral projections, GW] pipelines')
    plt.legend()
    plt.xlabel('Number of candidates allowed to 2nd stage')
    plt.ylabel('Classification accuracy')
    path = 'data/search_acc.jpg'#.format(name)    
    fig.savefig(path)
    print('fig saved to ', path)
    

def plot_cls_acc():
    '''
    Plot acc for classification. scatter plot for ot, gw, ot_gw
    '''
    plt.clf()
    
    ot_acc = [866, 866, 833, 783, 766, 800, 783, 700, 916, 750, 850, 866, 850, 800, 850]
    gw_acc = [900, 900, 950, 916, 833, 966, 850, 833, 966, 900, 966, 866, 883, 950, 883]
    combine_acc = [933, 933, 966, 950, 900, 983, 900, 850, 1000, 900, 983, 916, 900, 933, 933]
    x_l = range(1, len(gw_acc)+1) #[3, 5, 10, 15]
    ot_acc = np.array(ot_acc)
    gw_acc = np.array(gw_acc)
    combine_acc = np.array(combine_acc)
    idx = np.argsort(combine_acc)
    gw_acc = gw_acc[idx]
    combine_acc = combine_acc[idx]
    #pdb.set_trace()
    fig = plt.figure()
    #ax = ax
    #plt.plot(x_l, np.array(ot_acc)/1000, '-*', label='ot')
    plt.plot(x_l, np.array(combine_acc)/1000, '-o', label='COPT + GW combined')
    plt.plot(x_l, np.array(gw_acc)/1000, '-x', label='GW')
    
    plt.title('Classification acc of [GW] vs [COPT, GW]', fontsize=16)
    plt.legend(fontsize=13)
    plt.xlabel('Trial #', fontsize=15)
    plt.ylabel('Classification accuracy', fontsize=15)
    
    path = 'data/cls_acc.jpg'#.format(name)
    fig.savefig(path)
    print('fig saved to ', path)
    
def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
#create_dir(res_dir)
    
def plot_confusions():
    #plot_confusion(tgt, pred, labels=None, name=''):
    ot_cls = '0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 4., 1., 1., 4., 1., 1., 4., 1., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 9., 6., 9., 9., 9., 6., 6., 6., 9'
    gw_cls = '0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 6., 1., 1., 1., 1., 6., 6., 1., 6., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 6., 5., 6., 6., 6., 6., 5., 6., 6., 6., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9'
    combine_cls = '0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 4., 1., 1., 4., 1., 1., 1., 1., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 6., 5., 6., 6., 6., 6., 5., 6., 6., 6., 9., 9., 9., 9., 9., 9., 9., 9., 9., 9'
    ot_cls = parse_cls(ot_cls)
    gw_cls = parse_cls(gw_cls)
    combine_cls = parse_cls(combine_cls)
    
    tgt_cls = []
    for i in [0, 1, 4, 5, 6, 9]:
        tgt_cls.extend([i]*10)
    names = ['ot_cls', 'gw_cls', 'combine_cls']
    for i, pred in enumerate([ot_cls, gw_cls, combine_cls]):
        plot_confusion(tgt_cls, pred, name=names[i])

def plot_convergence():
    '''
    plot Convergence
    '''
    conv = '99.77512221820409, 47.0570797352225, 45.454712183248795, 43.46666639511483, 40.9052209134619, 34.46097733377226, 24.431260224317242, 23.51762172579995, 22.485459983772515, 22.106446259828203, 21.534230884767375, 21.463743674621867'
    ar = conv.split(', ')
    conv_ar = [float(f) for f in ar]
    x_l = [20*i for i in list(range(len(conv_ar)))]
    fig = plt.figure()
    #plt.errorbar(x_l, ot_acc, yerr=ot_std, marker='o', label='COPT sketches')
    plt.plot(x_l, conv_ar, '-o', label='COPT distance')
    #plt.errorbar(x_l, svd_acc, yerr=svd_std, marker='+', label='Spectral projections')
    
    plt.title('COPT distance convergence sketching a 50-node graph to 15 nodes')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('COPT distance')
    path = 'data/convergence.jpg'#.format(name)    
    fig.savefig(path)
    print('fig saved to ', path)
    
if __name__ == '__main__':
    """
    For testing utils functions.
    """
    '''
    n = 2
    L = create_graph_lap(n)
    view_graph(L, soft_edge=True)
    '''
    #plot_search_acc()
    plot_cls_acc()
    #plot_convergence()
    #plot_confusions()
    
