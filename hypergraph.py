
'''
using hypergraph representations for document classification.
'''
import _init_paths
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import utils
import math
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import sys
import time

import pdb

device = utils.device

class HyperMod(nn.Module):
    
    def __init__(self, input_dim, vidx, eidx, nv, ne, v_weight, e_weight, args, is_last=False, use_edge_lin=False):
        super(HyperMod, self).__init__()
        self.args = args
        #initialize representations
        #self.v = torch.randn( )
        #can use torch embedding layer!
        #self.v_embed = nn.Embedding(  )
        #self.e = torch.randn( )
        self.eidx = eidx
        self.vidx = vidx
        self.v_weight = v_weight
        self.e_weight = e_weight
        self.nv, self.ne = args.nv, args.ne

        
        self.W_v2e = Parameter(torch.randn(args.n_hidden, args.n_hidden))
        self.W_e2v = Parameter(torch.randn(args.n_hidden, args.n_hidden))
        self.b_v = Parameter(torch.zeros(args.n_hidden))
        self.b_e = Parameter(torch.zeros(args.n_hidden))
        self.is_last_mod = is_last
        self.use_edge_lin = use_edge_lin
        if is_last and self.use_edge_lin:
            self.edge_lin = torch.nn.Linear(args.n_hidden, args.final_edge_dim)
            
    def forward(self, v, e):
        #pdb.set_trace()
        #v = self.vtx_lin(v)
        if args.edge_linear:
            ve = torch.matmul(v, self.W_v2e) + self.b_v
        else:
            ve = F.relu(torch.matmul(v, self.W_v2e) + self.b_v)
        #weigh ve according to how many edges a vertex is connected to
        #ve *= self.v_weight
        #pdb.set_trace()
        v_fac = 4 if args.predict_edge else 1
        v = v*self.v_weight*v_fac #*3 *2
        #pdb.set_trace()        
        eidx = self.eidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        e = e.clone()
        ve = (ve*self.v_weight)[self.args.paper_author[:, 0]]
        ve *= args.v_reg_weight
        e.scatter_add_(src=ve, index=eidx, dim=0)
        e /= args.e_reg_sum
        #e = e*self.e_weight        
        ev = F.relu(torch.matmul(e, self.W_e2v) + self.b_e)
        #e = e*self.e_weight
        #ev *= self.e_weight
        #v = torch.zeros(self.nv , self.n_hidden)
        if False and self.args.cur_epoch % 4 == 0:
            print(self.W_v2e.grad)
            pdb.set_trace()
        #pdb.set_trace()
        vidx = self.vidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        ev_vtx = (ev*self.e_weight)[self.args.paper_author[:, 1]]
        #ev_vtx = (ev)[self.args.paper_author[:, 1]]
        ev_vtx *= args.e_reg_weight
        #v = v.clone()
        v.scatter_add_(src=ev_vtx, index=vidx, dim=0)
        #v = v*self.v_weight
        v /= args.v_reg_sum
        if not self.is_last_mod:
            v = F.dropout(v, args.dropout_p)
        if self.is_last_mod and self.use_edge_lin:
            ev_edge = (ev*torch.exp(self.e_weight)/np.exp(2))[self.args.paper_author[:, 1]]
            pdb.set_trace()
            v2 = torch.zeros_like(v)
            v2.scatter_add_(src=ev_edge, index=vidx, dim=0)
            v2 = self.edge_lin(v2)
            v = torch.cat([v, v2], -1)
        return v, e

    def forward00(self, v, e):
        #March normalization
        v = self.lin1(v)
        ve = F.relu(torch.matmul(v, self.W_v2e) + self.b_v)
        #weigh ve according to how many edges a vertex is connected to
        #ve *= self.v_weight
        v = v*self.v_weight #*2
        #pdb.set_trace()        
        eidx = self.eidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        e = e.clone()
        ve = (ve*self.v_weight)[self.args.paper_author[:, 0]]
        e.scatter_add_(src=ve, index=eidx, dim=0)
        
        ev = F.relu(torch.matmul(e, self.W_e2v) + self.b_e)
        e = e*self.e_weight
        #ev *= self.e_weight
        #v = torch.zeros(self.nv , self.n_hidden)
        if False and self.args.cur_epoch % 4 == 0:
            print(self.W_v2e.grad)
            pdb.set_trace()
        #pdb.set_trace()
        vidx = self.vidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        ev_vtx = (ev*self.e_weight)[self.args.paper_author[:, 1]]        
        #v = v.clone()
        v.scatter_add_(src=ev_vtx, index=vidx, dim=0)
        #v = v*self.v_weight
        if self.is_last_mod:
            ev_edge = (ev*torch.exp(self.e_weight)/np.exp(2))[self.args.paper_author[:, 1]]
            pdb.set_trace()
            v2 = torch.zeros_like(v)
            v2.scatter_add_(src=ev_edge, index=vidx, dim=0)
            v2 = self.edge_lin(v2)
            v = torch.cat([v, v2], -1)
        return v, e

class Hypergraph(nn.Module):
    '''
    Hypergraph class, uses weights for vertex-edge and edge-vertex incidence matrix.
    One large graph.
    '''
    def __init__(self, vidx, eidx, nv, ne, v_weight, e_weight, args):
        '''
        vidx: idx tensor of elements to select, shape (ne, max_n),
        shifted by 1 to account for 0th elem (which is 0)
        eidx has shape (nv, max n)..
        '''
        super(Hypergraph, self).__init__()
        self.args = args
        '''
        #initialize representations
        #self.v = torch.randn( )
        #can use torch embedding layer!
        #self.v_embed = nn.Embedding(  )
        #self.e = torch.randn( )
        self.eidx = eidx
        self.vidx = vidx
        self.v_weight = v_weight
        self.e_weight = e_weight
        self.nv, self.ne = args.nv, args.ne
        #torch.empty(m, n).uniform_(1,2) #torch.randn((m, n))
        self.W_v2e = Parameter(torch.randn(args.n_hidden, args.n_hidden))
        self.W_e2v = Parameter(torch.randn(args.n_hidden, args.n_hidden))
        self.b_v = Parameter(torch.zeros(args.n_hidden))
        self.b_e = Parameter(torch.zeros(args.n_hidden))
        '''
        self.hypermods = []
        is_first = True
        for i in range(args.n_layers):
            is_last = True if i == args.n_layers-1 else False            
            self.hypermods.append(HyperMod(args.input_dim if is_first else args.n_hidden, vidx, eidx, nv, ne, v_weight, e_weight, args, is_last=is_last))
            is_first = False

        if args.predict_edge:
            self.edge_lin = torch.nn.Linear(args.input_dim, args.n_hidden) #HERE

        self.vtx_lin = torch.nn.Linear(args.input_dim, args.n_hidden)
        #insetad of A have vector of indices
        #self.cls = nn.Linear(args.n_hidden+args.final_edge_dim, args.n_cls)
        self.cls = nn.Linear(args.n_hidden, args.n_cls)

    def to_device(self, device):
        self.to(device)
        for mod in self.hypermods:
            mod.to('cuda')
            #mod #HERE
        return self
        
    def all_params(self):
        params = []
        for mod in self.hypermods:
            params.extend(mod.parameters())
        return params
        
    def forward(self, v, e):
        '''
        Take initial embeddings from the select labeled data.
        Return predicted cls.
        '''
        v = self.vtx_lin(v)
        if self.args.predict_edge:
            e = self.edge_lin(e)
        for mod in self.hypermods:
            v, e = mod(v, e)
        #pdb.set_trace()
        pred = self.cls(v)
        return v, e, pred
        
class Hypertrain:
    def __init__(self, args):
        
        #cross entropy between predicted and actual labels
        self.loss_fn = nn.CrossEntropyLoss() #consider logits
        
        self.hypergraph = Hypergraph(args.vidx, args.eidx, args.nv, args.ne, args.v_weight, args.e_weight, args)
        #optim.Adam([self.P, self.Ly], lr=.4)
        self.optim = optim.Adam(self.hypergraph.all_params(), lr=.04)
        #'''
        milestones = [100*i for i in range(1, 4)] #[100, 200, 300]                                              
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=milestones, gamma=0.51)
        #'''
        self.args = args
        
    def train(self, v, e, label_idx, labels):
        self.hypergraph = self.hypergraph.to_device(device)
        #v = v.to(device)
        #e = e.to(device)
        #label_idx = label_idx.to(device)
        #labels = labels.to(device)
        v_init = v
        e_init = e
        best_err = sys.maxsize
        for i in range(self.args.n_epoch):
            args.cur_epoch = i
            
            v, e, pred_all = self.hypergraph(v_init, e_init)
            pred = pred_all[label_idx]            
            loss = self.loss_fn(pred, labels)
            if i % 30 == 0:
                test_err =self.eval(pred_all)
                if test_err < best_err:
                    best_err = test_err
            if i % 50 == 0:
                sys.stdout.write(' loss {} \t'.format(loss))
            self.optim.zero_grad()
            loss.backward()            
            self.optim.step()
            self.scheduler.step()
        #pdb.set_trace()
        e_loss = self.eval(pred_all)
        return pred_all, loss, best_err

    def eval(self, all_pred):

        if self.args.val_idx is None:
            ones = torch.ones(len(all_pred))
            ones[self.args.label_idx] = -1
        else:
            ones = -torch.ones(len(all_pred))
            ones[self.args.val_idx] = 1          
        
        #tgt = self.args.all_labels
        #tgt[self.args.label_idx] = -1
        
        tgt = self.args.all_labels
        tgt[ones==-1] = -1        
        fn = nn.CrossEntropyLoss(ignore_index=-1)
        loss = fn(all_pred, tgt)
        #print(' ~~ eval loss ~~ ', loss)
        
        #pdb.set_trace()
        tgt = self.args.all_labels[ones>-1]
        #tgt[self.args.label_idx] = -1
        pred = torch.argmax(all_pred, -1)[ones>-1]
        #pdb.set_trace()
        acc = torch.eq(pred, tgt).sum().item()/len(tgt)
        if args.verbose:
            print('TEST ERR ', 1-acc, ' ~~ eval loss ~~ ', loss)
        return 1-acc
        
def train(args):
    '''
    args.vidx, args.eidx, args.nv, args.ne, args = s
    args.e_weight = s
    args.v_weight = s
    label_idx, labels = s
    '''
    #args.e = torch.randn(args.ne, args.n_hidden)
    if args.predict_edge:
        args.e = args.edge_X
    else:
        args.e = torch.zeros(args.ne, args.n_hidden).to(device)
    #args.v = torch.randn(self.args.nv, args.n_hidden)            
    hypertrain = Hypertrain(args)

    pred_all, loss, test_err = hypertrain.train(args.v, args.e, args.label_idx, args.labels)
    return test_err
    
def gen_data_cora(args, data_path='data/cora_author.pt', flip_edge_node=True, do_val=False):
    '''
    Retrieve and process data, can be used generically for any dataset with predefined data format, eg cora, citeseer, etc.
    flip_edge_node: whether to flip edge and node in case of relation prediction.
    '''
    data_dict = torch.load(data_path)
    paper_author = torch.LongTensor(data_dict['paper_author'])
    author_paper = torch.LongTensor(data_dict['author_paper'])
    n_author = data_dict['n_author']
    n_paper = data_dict['n_paper']
    classes = data_dict['classes']
    #in sparse np array format
    paper_X = data_dict['paper_X']
    if args.predict_edge: #'author_X' in data_dict:
        #edge representations        
        author_X = data_dict['author_X']
        author_classes = data_dict['author_classes']
    paperwt = data_dict['paperwt']
    authorwt = data_dict['authorwt'] 
    #n_cls = data_dict['n_cls']
    cls_l = list(set(classes))
    
    #can flip nodes and edges here for e.g. learning hyperedge representations
    #can flip due to symmetry in HNHN
    if args.predict_edge:        
        if flip_edge_node:
            temp = paper_author
            paper_author = author_paper
            author_paper = temp
            temp = n_author
            n_author = n_paper
            n_paper = temp
            args.edge_X = torch.from_numpy(paper_X).to(torch.float32).to(device) #paper_X.to(device)
            args.edge_classes = classes
            temp = paper_X
            paper_X = author_X
            #author_X = temp
            classes = author_classes
            temp = paperwt
            paperwt = authorwt
            authorwt = temp
        else:            
            args.edge_X = torch.from_numpy(author_X).to(torch.float32).to(device)
            args.edge_classes = torch.LongTensor(author_classes).to(device)
    
    cls2int = {k:i for (i, k) in enumerate(cls_l)}
    classes = [cls2int[c] for c in classes]
    args.input_dim = paper_X.shape[-1] #300 if args.dataset_name == 'citeseer' else 300
    args.n_hidden = 800 if args.predict_edge else 400
    args.final_edge_dim = 100
    args.n_epoch = 140 if args.n_layers == 1 else 230 #130 #120
    args.ne = n_author
    args.nv = n_paper
    #args.n_layers = 1 #2 #2
    ne = args.ne
    nv = args.nv
    args.n_cls = len(cls_l)
    
    #no replacement!
    #n_labels = max(1, math.ceil(nv*.052))
    n_labels = max(1, math.ceil(nv*utils.get_label_percent(args.dataset_name)))
    args.all_labels = torch.LongTensor(classes)
    proportional_select = False
    if proportional_select:
        n_labels = int(math.ceil(n_labels/args.n_cls)*args.n_cls)
        all_cls_idx = []
        n_label_per_cls = n_labels//args.n_cls
        for i in range(args.n_cls):
            #pdb.set_trace()
            cur_idx = torch.LongTensor(list(range(nv)))[args.all_labels == i]
            rand_idx = torch.from_numpy(np.random.choice(len(cur_idx), size=(n_label_per_cls,), replace=False )).to(torch.int64)
            cur_idx = cur_idx[rand_idx]
            all_cls_idx.append(cur_idx)
        args.label_idx = torch.cat(all_cls_idx, 0)
    else:
        args.label_idx = torch.from_numpy(np.random.choice(nv, size=(n_labels,), replace=False )).to(torch.int64)    

    
    if do_val:
        #eg for cross validation on training set
        val_idx = torch.from_numpy(np.random.choice(len(args.label_idx), size=len(args.label_idx)//args.kfold ))
        args.val_idx = args.label_idx[val_idx]
        ones = torch.ones(len(args.label_idx))
        ones[val_idx] = -1
        args.label_idx = args.label_idx[ones>-1]
    else:
        args.val_idx = None
    
    args.labels = args.all_labels[args.label_idx].to(device) #torch.ones(n_labels, dtype=torch.int64)
    args.all_labels = args.all_labels.to(device)
    #pdb.set_trace()
    
    #isinstance(paper_X, scipy.sparse.csr.csr_matrix)
    if isinstance(paper_X, np.ndarray):
        args.v = torch.from_numpy(paper_X.astype(np.float32)).to(device)
    else:
        args.v = torch.from_numpy(np.array(paper_X.astype(np.float32).todense())).to(device)
        
    #labeled 
    #args.vidx, args.eidx, args.nv, args.ne, args.v_weight, args.e_weight
    #vidx has shape (ne, max n)
    #generate edges
    #edge weights. ensure labels.
    #args.vidx = torch.zeros((ne+1,), dtype=torch.int64).random_(0, nv-1) + 1 #np.random.randint(nv, (ne, 3))
    #args.eidx = torch.zeros((nv+1,), dtype=torch.int64).random_(0, ne-1) + 1 #torch.random.randint(ne, (nv, 2))
    args.vidx = paper_author[:, 0].to(device)
    args.eidx = paper_author[:, 1].to(device)
    args.paper_author = paper_author
    #pdb.set_trace()
    args.v_weight = torch.Tensor([(1/w if w > 0 else 1) for w in paperwt]).unsqueeze(-1).to(device) #torch.ones((nv, 1)) / 2 #####
    args.e_weight = torch.Tensor([(1/w if w > 0 else 1) for w in authorwt]).unsqueeze(-1).to(device) # 1)) / 2 #####torch.ones(ne, 1) / 3
    assert len(args.v_weight) == nv and len(args.e_weight) == ne

    #args.alpha = 0.15 #.1 #-.1
    #
    #weights for regularization
    #'''
    paper2sum = defaultdict(list)
    author2sum = defaultdict(list)
    e_reg_weight = torch.zeros(len(paper_author)) ###
    v_reg_weight = torch.zeros(len(paper_author)) ###
    #a switch to determine whether to have wt in exponent or base
    use_exp_wt = args.use_exp_wt #True #False
    for i, (paper_idx, author_idx) in enumerate(paper_author.tolist()):
        e_wt = args.e_weight[author_idx]
        e_reg_wt = torch.exp(args.alpha_e*e_wt) if use_exp_wt else e_wt**args.alpha_e 
        e_reg_weight[i] = e_reg_wt
        paper2sum[paper_idx].append(e_reg_wt) ###
        
        v_wt = args.v_weight[paper_idx]
        v_reg_wt = torch.exp(args.alpha_v*v_wt) if use_exp_wt else v_wt**args.alpha_v
        v_reg_weight[i] = v_reg_wt
        author2sum[author_idx].append(v_reg_wt) ###        
    #'''
    v_reg_sum = torch.zeros(nv) ###
    e_reg_sum = torch.zeros(ne) ###
    for paper_idx, wt_l in paper2sum.items():
        v_reg_sum[paper_idx] = sum(wt_l)
    for author_idx, wt_l in author2sum.items():
        e_reg_sum[author_idx] = sum(wt_l)

    pdb.set_trace()
    #this is used in denominator only
    e_reg_sum[e_reg_sum==0] = 1
    v_reg_sum[v_reg_sum==0] = 1
    args.e_reg_weight = torch.Tensor(e_reg_weight).unsqueeze(-1).to(device)
    args.v_reg_sum = torch.Tensor(v_reg_sum).unsqueeze(-1).to(device)
    args.v_reg_weight = torch.Tensor(v_reg_weight).unsqueeze(-1).to(device)
    args.e_reg_sum = torch.Tensor(e_reg_sum).unsqueeze(-1).to(device)
    return args

def gen_data_dblp(args, data_path='data/dblp_data.pt', do_val=False):
    """
    Exact same data as generated by hypergcn.
    """
    #paper_idx':paper_idx, 'author':author_idx, 'paper_X':dataset['features'], 'train_idx':train, 'test_idx':test
    data = torch.load(data_path)
    ###
    #pdb.set_trace()

    args.n_hidden = 800 if args.dataset_name == 'pubmed' else 400
    args.final_edge_dim = 100
    args.n_epoch = 200
    #args.n_layers = 1 #2
    
    paperwt = data['paperwt']
    authorwt = data['authorwt']
    train_idx = torch.LongTensor(data['train_idx'])
    
    args.label_idx = torch.from_numpy(np.random.choice(len(paperwt), size=len(train_idx) ))
    if do_val:
        #eg for cross validation on training set
        val_idx = torch.from_numpy(np.random.choice(len(args.label_idx), size=len(args.label_idx)//args.kfold ))
        args.val_idx = args.label_idx[val_idx]
        ones = torch.ones(len(train_idx))
        ones[val_idx] = -1
        args.label_idx = args.label_idx[ones>-1]
    else:
        args.val_idx = None
        
    args.all_labels = torch.from_numpy(np.where(data['labels'])[1]).to(torch.int64).to(device)
    args.labels = args.all_labels[args.label_idx].to(device)
    
    if args.do_svd:
        svd = TruncatedSVD(n_components=300, n_iter=7)
        X = svd.fit_transform(data['paper_X'])
    else:
        X = data['paper_X']
    args.input_dim = X.shape[-1] #300 ###
    args.v = torch.from_numpy(X).to(device)
    args.n_cls = int(args.all_labels.max()) + 1 #len(cls_l)
    #pdb.set_trace()
    '''
    #no replacement!
    n_labels = max(1, int(nv*.052))
    args.all_labels = torch.LongTensor(classes)
    if False:
        n_labels = int(math.ceil(n_labels/args.n_cls)*args.n_cls)
        all_cls_idx = []
        n_label_per_cls = n_labels//args.n_cls
        for i in range(args.n_cls):
            #pdb.set_trace()
            cur_idx = torch.LongTensor(list(range(nv)))[args.all_labels == i]
            rand_idx = torch.from_numpy(np.random.choice(len(cur_idx), size=(n_label_per_cls,), replace=False )).to(torch.int64)
            cur_idx = cur_idx[rand_idx]
            all_cls_idx.append(cur_idx)
        args.label_idx = torch.cat(all_cls_idx, 0)
    else:
        args.label_idx = torch.from_numpy(np.random.choice(nv, size=(n_labels,), replace=False )).to(torch.int64)    
        
    args.labels = args.all_labels[args.label_idx] #torch.ones(n_labels, dtype=torch.int64)
    #args.v = torch.from_numpy(paper_X.astype(np.float32))        
    '''
    ###
    args.ne = len(authorwt)
    args.nv = len(paperwt)
    ne = args.ne
    nv = args.nv
    
    args.vidx = torch.from_numpy(data['paper_idx']).to(torch.int64).to(device) # paper_author[:, 0]
    #pdb.set_trace()
    if 'author' in data:
        data['author_idx'] = data['author']
    args.eidx = torch.from_numpy(data['author_idx']).to(torch.int64).to(device) #paper_author[:, 1]
    #pdb.set_trace()
    args.paper_author = torch.stack([args.vidx, args.eidx], -1)
    paper_author = args.paper_author
    args.v_weight = torch.Tensor([(1/w if w > 0 else 1) for w in paperwt]).unsqueeze(-1).to(device) #torch.ones((nv, 1)) / 2 #####
    args.e_weight = torch.Tensor([(1/w if w > 0 else 1) for w in authorwt]).unsqueeze(-1).to(device) # 1)) / 2 #####torch.ones(ne, 1) / 3
    
    #weights for regularization    
    #v_reg_wt = torch.exp(args.alpha*v_wt) if use_exp_wt else v_wt**args.alpha 
    #v_reg_weight[i] = v_reg_wt
    paper2sum = defaultdict(list)
    author2sum = defaultdict(list)
    e_reg_weight = torch.zeros(len(paper_author)) ###
    v_reg_weight = torch.zeros(len(paper_author)) ###
    #a switch to determine whether to have wt in exponent or base
    use_exp_wt = args.use_exp_wt #True #False
    for i, (paper_idx, author_idx) in enumerate(paper_author.tolist()):
        e_wt = args.e_weight[author_idx]
        e_reg_wt = torch.exp(args.alpha_e*e_wt) if use_exp_wt else e_wt**args.alpha_e 
        e_reg_weight[i] = e_reg_wt
        paper2sum[paper_idx].append(e_reg_wt) ###
        
        v_wt = args.v_weight[paper_idx]
        v_reg_wt = torch.exp(args.alpha_v*v_wt) if use_exp_wt else v_wt**args.alpha_v
        v_reg_weight[i] = v_reg_wt
        author2sum[author_idx].append(v_reg_wt) ###
        
    pdb.set_trace()
    #'''
    v_reg_sum = torch.zeros(nv) ###
    e_reg_sum = torch.zeros(ne) ###
    for paper_idx, wt_l in paper2sum.items():
        v_reg_sum[paper_idx] = sum(wt_l)
    for author_idx, wt_l in author2sum.items():
        e_reg_sum[author_idx] = sum(wt_l)    
    
    #this is used in denominator only
    e_reg_sum[e_reg_sum==0] = 1
    v_reg_sum[v_reg_sum==0] = 1

    args.e_reg_weight = torch.Tensor(e_reg_weight).unsqueeze(-1).to(device)
    args.v_reg_sum = torch.Tensor(v_reg_sum).unsqueeze(-1).to(device)
    args.v_reg_weight = torch.Tensor(v_reg_weight).unsqueeze(-1).to(device)
    args.e_reg_sum = torch.Tensor(e_reg_sum).unsqueeze(-1).to(device)
    #pdb.set_trace()
    '''
    paper2sum = defaultdict(list)
    e_reg_weight = torch.zeros() ###
    for paper_idx, author_idx in paper_author:
        wt = args.e_weight[author_idx]
        reg_wt = torch.exp(args.alpha*wt)
        e_reg_weight[author_idx] = reg_wt
        paper2sum[paper_idx].append(reg_wt) ###
    '''
    
    return args

def gen_synthetic_data(args):
    '''
    Generate synthetic data. 
    '''
    args.n_hidden = 50
    args.n_epoch = 200
    args.ne = 2
    args.nv = 3
    ne = args.ne
    nv = args.nv    
    #no replacement!
    n_labels = max(1, int(nv*.1))
    #numpy.random.choice(a, size=None, replace=True, p=None)
    args.label_idx = torch.from_numpy(np.random.choice(nv, size=(n_labels,), replace=False )).to(torch.int64)
    #args.label_idx = torch.zeros(n_labels, dtype=torch.int64).random_(0, nv) #torch.randint(torch.int64, nv, (int(nv*.1), ))
    args.labels = torch.ones(n_labels, dtype=torch.int64)
    args.labels[:n_labels//2] = 0
    args.n_cls = 2
    #labeled 
    #args.vidx, args.eidx, args.nv, args.ne, args.v_weight, args.e_weight
    #vidx has shape (ne, max n)
    #generate edges
    #check upper bound!
    #args.vidx = torch.zeros((ne+1, 3), dtype=torch.int64).random_(0, nv) + 1 #np.random.randint(nv, (ne, 3))
    #args.eidx = torch.zeros((nv+1, 2), dtype=torch.int64).random_(0, ne) + 1 #torch.random.randint(ne, (nv, 2))
    args.vidx = torch.zeros((ne,), dtype=torch.int64).random_(0, nv-1) + 1 #np.random.randint(nv, (ne, 3))
    args.eidx = torch.zeros((nv,), dtype=torch.int64).random_(0, ne-1) + 1 #torch.random.randint(ne, (nv, 2))    
    args.v_weight = torch.ones((nv, 1)) / 2
    args.e_weight = torch.ones(ne, 1) / 3
    train(args)

    
def compare_normalization(data_path, args):
    """
    Studying the effects of normalization hyperparameters on test accuracy.
    """
    best_err = sys.maxsize
    best_err_std = sys.maxsize
    best_alpha_v = sys.maxsize
    best_alpha_e = sys.maxsize
    print('ARGS {}'.format(args))
    mean_err_l = []
    mean_err_std_l = []
    time_l = []
    time_std_l = []
    n_runs = 1 #5
    n_runs = 3 #5
    a_list = [0] #
    a_list = range(-2, 2, 1)
    #a_list = range(-20, -5, 3)
    a_list = range(-1, 1, 1)
    a_list1 = []
    test_alpha = False #True
    for av in a_list:
        if test_alpha:
            #a_list1.append([av, 0]) #test beta
            a_list1.append([0, av]) #test alpha
        else:
            for ae in a_list:
                a_list1.append([av, ae])
    sys.stdout.write('alpha beta list {}'.format(a_list1))
    for av, ae in a_list1:
        args.alpha_v = av/10
        args.alpha_e = ae/10
        print('ALPHA ', args.alpha_v)
        err_ar = np.zeros(n_runs)
        time_ar = np.zeros(n_runs)
        for i in range(n_runs):
            if args.dataset_name in ['dblp', 'pubmed']:                
                data_path = 'data/pubmed_data.pt' if args.dataset_name == 'pubmed' else 'data/dblp_data.pt'
                args = gen_data_dblp(args, data_path=data_path)
            else:
                args = gen_data_cora(args, data_path=data_path)
                
            #pred_all, loss, test_err = hypertrain.train(args.v, args.e, args.label_idx, args.labels)
            time0 = time.time()
            test_err = train(args)
            time_ar[i] = time.time() - time0
            err_ar[i] = test_err
            sys.stdout.write(' test err {}\t'.format(test_err))
        mean_err = err_ar.mean()
        err_std = err_ar.std()
        mean_err_l.append(mean_err)
        mean_err_std_l.append(err_std)
        dur = time_ar.mean()
        time_l.append(dur)
        time_std_l.append(time_ar.std())
        
        sys.stdout.write('\n ~~~Mean test err {}+-{} for alpha {} {} time {}~~~\n'.format(np.round(mean_err, 2), np.round(err_std, 2), args.alpha_v, args.alpha_e, dur ))
        if mean_err < best_err:
            best_err = mean_err
            best_err_std = err_std
            best_alpha_v = args.alpha_v
            best_alpha_e = args.alpha_e
            best_time = np.round(dur, 3)
            best_time_std = time_ar.std()
    print('mean errs {} mean err std {}'.format(mean_err_l, mean_err_std_l))
    print('best err {}+-{} best alpha_v {} alpha_e {} for dataset {}'.format(np.round(best_err*100, 2), np.round(best_err_std*100, 2), best_alpha_v, best_alpha_e, args.dataset_name))
    print('best ACC {}+-{} time {}+-{}'.format(np.round((1-best_err)*100, 2), np.round(best_err_std*100, 2), best_time, best_time_std  ))
    #train(args)

def select_params(data_path, args):
    #find best hyperparameters with by splitting training set into train + validation set
    best_err = sys.maxsize
    best_err_std = sys.maxsize
    best_alpha_v = sys.maxsize
    best_alpha_e = sys.maxsize
    print('ARGS {}'.format(args))
    mean_err_l = []
    mean_err_std_l = []
    time_l = []
    time_std_l = []
    args.kfold = 1 #5
    args.kfold = 5 #5
    kfold = args.kfold
    a_list = [0] #
    a_list = range(-2, 2, 1)
    #a_list = range(-20, -5, 3)
    a_list = range(-1, 2, 1)
    a_list1 = []
    test_alpha = False #True
    for av in a_list:
        if test_alpha:
            #a_list1.append([av, 0]) #test beta
            a_list1.append([0, av]) #test alpha
        else:
            for ae in a_list:
                a_list1.append([av, ae])
    sys.stdout.write('alpha beta list {}'.format(a_list1))
    for av, ae in a_list1:
        args.alpha_v = av/10
        args.alpha_e = ae/10
        print('ALPHA ', args.alpha_v)
        err_ar = np.zeros(kfold)
        time_ar = np.zeros(kfold)
        for i in range(kfold):
            if args.dataset_name in ['dblp', 'pubmed']:                
                data_path = 'data/pubmed_data.pt' if args.dataset_name == 'pubmed' else 'data/dblp_data.pt'
                args = gen_data_dblp(args, data_path=data_path, do_val=True)
            else:
                args = gen_data_cora(args, data_path=data_path, do_val=True)
                
            #pred_all, loss, test_err = hypertrain.train(args.v, args.e, args.label_idx, args.labels)
            time0 = time.time()
            test_err = train(args)
            time_ar[i] = time.time() - time0
            err_ar[i] = test_err
            sys.stdout.write(' Validation err {}\t'.format(test_err))
        mean_err = err_ar.mean()
        err_std = err_ar.std()
        mean_err_l.append(mean_err)
        mean_err_std_l.append(err_std)
        dur = time_ar.mean()
        time_l.append(dur)
        time_std_l.append(time_ar.std())
        
        sys.stdout.write('\n ~~~Mean VAL err {}+-{} for alpha {} {} time {}~~~\n'.format(np.round(mean_err, 2), np.round(err_std, 2), args.alpha_v, args.alpha_e, dur ))
        if mean_err < best_err:
            best_err = mean_err
            best_err_std = err_std
            best_alpha_v = args.alpha_v
            best_alpha_e = args.alpha_e
            best_time = np.round(dur, 3)
            best_time_std = time_ar.std()
    print('mean validation errs {} mean err std {}'.format(mean_err_l, mean_err_std_l))
    print('best err {}+-{} best alpha_v {} alpha_e {} for dataset {}'.format(np.round(best_err*100, 2), np.round(best_err_std*100, 2), best_alpha_v, best_alpha_e, args.dataset_name))
    print('best validation ACC {}+-{} time {}+-{}'.format(np.round((1-best_err)*100, 2), np.round(best_err_std*100, 2), best_time, best_time_std  ))
    return best_alpha_v, best_alpha_e

    
if __name__ =='__main__':
    args = utils.parse_args()
    dataset_name = args.dataset_name #'citeseer' #'cora'
    data_path = None
    if dataset_name == 'cora':
        if args.do_svd:
            data_path = 'data/cora_author_10cls300.pt'
        else:
            data_path = 'data/cora_author_10cls1000.pt'
    elif dataset_name == 'citeseer':
        if args.do_svd:
            data_path = 'data/citeseer.pt'
        else:
            data_path = 'data/citeseer6cls3703.pt'            
    elif dataset_name not in ['dblp', 'pubmed']:
        #args = gen_data_dl(args)
        raise Exception('dataset {} not supported!'.format(dataset_name))

    if args.fix_seed:
        np.random.seed(0)
        torch.manual_seed(0)
        
    #if doing cross validate
    do_cross_val = True
    if do_cross_val:
        select_params(data_path, args)

    #if studying the effects of hyperparameters
    study_normalization = False
    if study_normalization:
        compare_normalization(data_path, args)
