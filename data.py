
'''
Processing data
'''
import torch
import numpy as np
import os
from collections import defaultdict
import re
import sklearn
import sklearn.feature_extraction as feat_extract
from sklearn.decomposition import TruncatedSVD
import utils

import pdb


def process_citeseer(args):
    '''
    citeseer.content of form <paper_id> <word_attributes>+ <class_label>
    citeseer.cites of form <id of cited paper> <id of citing paper>
    paper_citing, cited papers are hypernodes, citing papers are hyperedges.
    '''
    content_path = 'data/citeseer/citeseer.content'
    cites_path = 'data/citeseer/citeseer.cites'
    paper2citing = defaultdict(set)
    citing2paper = defaultdict(set)
    with open(cites_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            try:
                #paper2citing[int(line[0])].add(int(line[1]))
                #citing2paper[int(line[1])].add(int(line[0]))
                paper2citing[line[0]].add(line[1])
                citing2paper[line[1]].add(line[0])                
            except ValueError as e:
                pdb.set_trace()
                continue
            
    id2paper_idx = {}
    id2citing_idx = {}
    paper_citing = []
    citing_paper = []
    paperwt = torch.zeros(len(paper2citing))
    citingwt = torch.zeros(len(paper2citing))
    n_paper = 0
    for citing, papers in citing2paper.items():
        #pdb.set_trace()
        #remove papers that are only cited by one paper
        if len(papers) == 1:
            continue
        #pdb.set_trace()
        citing_idx = id2citing_idx[citing] if citing in id2citing_idx else len(id2citing_idx)
        id2citing_idx[citing] = citing_idx
        for paper in papers:
            paper_idx = id2paper_idx[paper] if paper in id2paper_idx else len(id2paper_idx)
            id2paper_idx[paper] = paper_idx
            paper_citing.append([paper_idx, citing_idx])
            citing_paper.append([citing_idx, paper_idx])
            citingwt[citing_idx] += 1
            paperwt[paper_idx] += 1
    n_paper = len(id2paper_idx.keys())
    n_citing = len(id2citing_idx.keys())
    paperwt = paperwt[:n_paper]
    citingwt = citingwt[:n_citing]
    X = np.zeros((n_paper, 3703))
    citing_X = np.zeros((n_citing, 3703))
    classes = ['ML'] * n_paper
    citing_classes = ['ML'] * n_citing
    n_skips = 0
    paper_idx_set = set()
    with open(content_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            
            paper_id = line[0]
            x = torch.FloatTensor([int(w) for w in line[1:-1]])
            if paper_id in id2paper_idx:
                idx = id2paper_idx[paper_id]
                paper_idx_set.add(idx)
                X[idx] = x
                classes[idx] = line[-1]
            elif paper_id in id2citing_idx:
                idx = id2citing_idx[paper_id]
                paper_idx_set.add(idx)
                citing_X[idx] = x
                citing_classes[idx] = line[-1]
            else:
                #count skips
                n_skips += 1                
                                
        print('number of papers skipped {}'.format(n_skips))
        print('paper idx set {}'.format(len(paper_idx_set)))

    feat_dim = X.shape[-1]
    do_svd = args.do_svd #False
    if do_svd:
        feat_dim = 300
        svd = TruncatedSVD(n_components=feat_dim, n_iter=12)
        X = svd.fit_transform(X)
        
    print('total number of data points/papers {}'.format(len(X)))
    cls2idx = {}
    for cls in set(classes):
        cls2idx[cls] = cls2idx[cls] if cls in cls2idx else len(cls2idx)
    classes = [cls2idx[c] for c in classes]
    citing_classes = [cls2idx[c] for c in citing_classes]
    pdb.set_trace()
    torch.save({'n_author': n_citing, 'n_paper': n_paper, 'classes':classes, 'author_classes': citing_classes, 'paper_author': paper_citing, 'author_paper': citing_paper, 'paperwt': paperwt, 'authorwt': citingwt, 'paper_X': X, 'author_X': citing_X}, 'data/citeseer{}cls{}.pt'.format(len(set(classes)), feat_dim) ) #_{}cls.pt'.format(len(set(classes))))

def process_citeseer_edge():
    '''
    Processing Citeseer data including the edges. Include 
    citeseer.content of form <paper_id> <word_attributes>+ <class_label>
    citeseer.cites of form <id of cited paper> <id of citing paper>
    paper_citing, cited papers are hypernodes, citing papers are hyperedges.
    '''
    content_path = 'data/citeseer/citeseer.content'
    cites_path = 'data/citeseer/citeseer.cites'
    paper2citing = defaultdict(set)
    citing2paper = defaultdict(set)
    with open(cites_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            try:
                #paper2citing[int(line[0])].add(int(line[1]))
                #citing2paper[int(line[1])].add(int(line[0]))
                paper2citing[line[0]].add(line[1])
                citing2paper[line[1]].add(line[0])                
            except ValueError as e:
                pdb.set_trace()
                continue
            
    id2paper_idx = {}
    id2citing_idx = {}
    paper_citing = []
    citing_paper = []
    paperwt = torch.zeros(len(paper2citing))
    citingwt = torch.zeros(len(paper2citing))
    n_paper = 0
    for citing, papers in citing2paper.items():
        #pdb.set_trace()
        #remove papers that are only cited by one paper
        if len(papers) == 1:
            continue
        #pdb.set_trace()
        citing_idx = id2citing_idx[citing] if citing in id2citing_idx else len(id2citing_idx)
        id2citing_idx[citing] = citing_idx
        for paper in papers:
            paper_idx = id2paper_idx[paper] if paper in id2paper_idx else len(id2paper_idx)
            id2paper_idx[paper] = paper_idx
            paper_citing.append([paper_idx, citing_idx])
            citing_paper.append([citing_idx, paper_idx])
            citingwt[citing_idx] += 1
            paperwt[paper_idx] += 1
    n_paper = len(id2paper_idx.keys())
    n_citing = len(id2citing_idx.keys())
    paperwt = paperwt[:n_paper]
    citingwt = citingwt[:n_citing]
    X = np.zeros((n_paper, 3703))
    classes = ['ML'] * n_paper
    n_skips = 0
    paper_idx_set = set()
    with open(content_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            
            paper_id = line[0]
            if paper_id not in id2paper_idx:
                #count skips
                n_skips += 1                
                continue
                #raise Exception('not in id2')            
            x = torch.FloatTensor([int(w) for w in line[1:-1]])
            idx = id2paper_idx[paper_id]
            paper_idx_set.add(idx)
            X[idx] = x
            classes[idx] = line[-1]
        print('number of papers skipped {}'.format(n_skips))
        print('paper idx set {}'.format(len(paper_idx_set)))

    feat_dim = X.shape[-1]
    do_svd = False
    if do_svd:
        feat_dim = 300
        svd = TruncatedSVD(n_components=feat_dim, n_iter=12)
        X = svd.fit_transform(X)
        
    print('total number of data points/papers {}'.format(len(X)))
    cls2idx = {}
    for cls in set(classes):
        cls2idx[cls] = cls2idx[cls] if cls in cls2idx else len(cls2idx)
    classes = [cls2idx[c] for c in classes]
    pdb.set_trace()
    torch.save({'n_author': n_citing, 'n_paper': n_paper, 'classes':classes, 'paper_author': paper_citing, 'author_paper': citing_paper, 'paperwt': paperwt, 'authorwt': citingwt, 'paper_X': X}, 'data/citeseer{}.pt'.format(feat_dim) ) #_{}cls.pt'.format(len(set(classes))))

    
'''
Example author line:
Title: DESIGN SPACE NAVIGATION AS A COLLABORATIVE AID
Author: CHARLES PETRIE AND MARK CUTKOSKY AND HISUP PARK
Abstract: The Redux 0 server is an agent, with 
For cora, authors are hyperedges and papers are edges.
'''
a_pat = re.compile("\s*and\s*|\s*,\s*")

def process_meta_files(args, path="data/cora/extractions"):
    '''
    process directory of meta files. Want tuples of indices of which author 
    wrote which papers, and which papers were written by which authors.
    Authors and papers designated by integer indices. 
    Abstract: The Redux 0 server is an agent, with 
    '''
    author2idx = {}
    paper2idx = {}
    paper_author = []
    author_paper = []    
    author_cnt = defaultdict(int)
    n_author = 0
    n_paper = 0      
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) ]
    #files = files[:10000]
    #files = files[:2000]
    paper2cls = torch.load('data/paper2cls.pt') #process_cora_cls()
    
    #keep top 7 classes
    n_cls = 10 #15 #7
    cls2cnt = defaultdict(int)
    for fi in files:
        fname = fi.split('/')[-1]
        if fname not in paper2cls:
            continue
        cur_cls = paper2cls[fname]
        cls2cnt[cur_cls] += 1
    classes = list(cls2cnt.keys())
    classes.sort(key=lambda x : cls2cnt[x] )
    cls_set = set(classes[-n_cls:])
    #pdb.set_trace()    
    
    #paper2cls = process_cora_cls()
    #n_cls = len(set(paper2cls.values()))
    author_lines = []
    paper_lines = []
    fname_lines = []
    abstract_lines = []
    #Title: The Redux 0 S
    title_lines = []
    classes = []
    empty_cnt = 0
    #papers and authors to weights, determined by freq
    paper2wt = defaultdict(set)
    author2wt = defaultdict(set)
    for fi in files:
        with open(fi) as f:
            try:
                lines = f.readlines()
            except UnicodeDecodeError as e:
                print('UnicodeDecodeError! ', e)
                continue
        a_line, p_line, ab_line = None, None, None
        for line in lines:
            if line[:7] == 'Title: ':
                p_line = line[7:].lower().strip()
            elif line[:8] == 'Author: ':
                a_line = line[8:].lower().strip()
            elif line[:10] == 'Abstract: ':
                ab_line = line[10:].lower().strip()

        fname = fi.split('/')[-1]
        if a_line is None or p_line is None or ab_line is None or fname not in paper2cls or paper2cls[fname] not in cls_set:
            #print('author line blank!')
            #pdb.set_trace()
            empty_cnt += 1
            continue
        author_lines.append(a_line)
        paper_lines.append(p_line)
        abstract_lines.append(ab_line)
        #title_lines.append(t_line) #add title!
        fname_lines.append(fname)
        #paper_cnt = defaultdict(int)
        authors = a_pat.split( a_line)        
        for author in authors:
            author_cnt[author] += 1
        #collect word freq, for tfidf
    #stop words!!
    tfidf = feat_extract.text.TfidfVectorizer(max_features=1000, ngram_range=(1,2), max_df=.2)
    sel_idx = []
    
    for i, a_line in enumerate(author_lines):
        authors = a_pat.split(a_line) #a_line.split(' and ')
        p_line = paper_lines[i]
        f_line = fname_lines[i]
        if f_line not in paper2cls:
            #print('not in dict')
            #pdb.set_trace()
            continue
        cur_cls = paper2cls[f_line]
        '''
        Title: DESIGN SPACE NAVIGATION AS A COLLABORATIVE AID
        Author: CHARLES PETRIE AND MARK CUTKOSKY AND HISUP PARK
        '''                        
        #for author in a:
        a_sum = [author_cnt[a] for a in authors]
        
        if sum(a_sum) <= len(a_sum):
            #only keep papers where at least one author has two papers.:
            continue
        
        paper = p_line
        '''
        if paper in paper2idx:
            paper_idx = paper2idx[paper]
        else:
            paper_idx = len(paper2idx)
            paper2idx[paper] = paper_idx
        '''
        paper2idx[paper] = n_paper
        paper_idx = n_paper
        n_paper += 1
        classes.append(cur_cls)
        sel_idx.append(i)
        for author in authors:
            #only keep the 
            if author_cnt[author] == 1:
                continue
            
            if author in author2idx:
                author_idx = author2idx[author]
            else:
                author_idx = len(author2idx)
                author2idx[author] = author_idx
            paper2wt[n_paper-1].add(author_idx)
            author2wt[author_idx].add(n_paper-1)
            paper_author.append([paper_idx, author_idx])
            author_paper.append([author_idx, paper_idx])

    #tfidf.fit([abstract_lines[i] for i in sel_idx])
    X = tfidf.fit_transform([abstract_lines[i] for i in sel_idx])
    print('created abstrct rep!')
    feat_dim = X.shape[-1]
    do_svd = args.do_svd #False
    if do_svd:
        feat_dim = 300
        svd = TruncatedSVD(n_components=feat_dim, n_iter=12)
        X = svd.fit_transform(X)
        #svd = TruncatedSVD(n_components=300, n_iter=7)
        #X = svd.fit_transform(X)
    else:
        X = X.todense()

    #pdb.set_trace()
    
    #author embed?! abstracts of authors!!
    
    n_author = len(author2idx)
    paperwt = torch.zeros(n_paper)
    authorwt = torch.zeros(n_author)
    
    for k in paper2wt.keys():
        paperwt[k] = len(paper2wt[k])        
    for k in author2wt.keys():
        authorwt[k] = len(author2wt[k])
            
    print('empty count ', empty_cnt)    
    torch.save({'n_author': n_author, 'n_paper': n_paper, 'classes':classes, 'paper_author': paper_author, 'author_paper': author_paper, 'paperwt': paperwt, 'authorwt': authorwt, 'paper_X': X}, 'data/cora_author_{}cls{}.pt'.format(len(set(classes)), feat_dim ))
    pdb.set_trace()
    return paper_author, author_paper, n_author, n_paper

def process_cora_cls(args, path='data/cora/classifications'):
    '''
    classification for cora papers
    '''
    papers = set()
    classes = set()
    paper2cls = {}
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if len(line) < 2:   
                #pdb.set_trace()
                continue
            #elif len(line) > 2:
            #    pdb.set_trace()
            #assert len(line) == 2
            paper = line[0]
            cls = line[1].split('/')[1]
            papers.add(line[0])
            classes.add(line[1])            
            paper2cls[paper] = cls
            
    print('cora number of lines {} papers {} classes {}'.format(len(lines), len(papers), len(classes)))
    #pdb.set_trace()
    torch.save(paper2cls, 'data/paper2cls.pt')
    return paper2cls
    
if __name__ == '__main__':
    args = utils.parse_args()
    dataset_name = 'citeseer' #'cora'
    dataset_name = 'cora'
    if dataset_name == 'cora':
        #process_cora_cls('data/cora/classifications')
        process_meta_files(args)
    else:
        process_citeseer(args)
