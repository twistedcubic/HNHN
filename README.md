
# Hypergraph Networks with Hyperedge Neurons (HNHN)

This repo contains the companion code for our paper:

[**HNHN: Hypergraph Networks with Hyperedge Neurons**](https://arxiv.org/abs/2006.12278)<br>
By [Yihe Dong](https://yihedong.me/), [Will Sawin](https://williamsawin.com/), and [Yoshua Bengio](https://yoshuabengio.org/)

One can run training on the main HNHN script as follows:

E.g. specify dataset name, set random seed, and number of layers:

`python hypergraph.py --dataset_name citeseer --seed --n_layers 1`

E.g. use a dimension-reduced version of the dataset:

`python hypergraph.py --dataset_name cora --do_svd`

The functions that call on baseline repository codes are in baseline.py, and can be called as follows:

`python baseline.py --dataset_name citeseer --method hgnn`

For all runtime options, please see either utils.py or run:

`python hypergraph.py --h`

Note:
[`baselines.py`](baselines.py) assumes the presence of two baseline repos: [HyperGCN](https://github.com/malllabiisc/HyperGCN) and [HGNN](https://github.com/iMoonLab/HGNN). If one wishes to run these baseline models, their directories should be named as `hypergcn` and `hgnn`, respectively, and should be placed in the parent directory of the current repo, as indicated in the [_init_paths.py script](_init_paths.py).


## Data processing

[data.py](data.py) extracts and processes raw data as described in the paper. For instance the [Cora Information Extraction data](https://people.cs.umass.edu/mccallum/data.html).

For an example of processed data please see the [CiteSeer data](data/citeseer6cls3703.pt) ([citeseer.pt](data/citeseer.pt) contains the same hypergraph but with reduced feature dimension).

p.s. please tag me when creating an issue, GitHub currrently cannot notify me of new issues.