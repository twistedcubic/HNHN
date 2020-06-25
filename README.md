
# Hypergraph Networks with Hyperedge Neurons (HNHN)

One can run training on the main HNHN script as follows:

E.g. specify dataset name, set random seed, and number of layers: `python hypergraph.py --dataset_name dblp --seed --n_layers 2`
E.g. use a dimension-reduced version of the dataset: `python hypergraph.py --dataset_name cora --do_svd`

The functions that call on baseline repository codes are in baseline.py, and can be called as follows:

`python baseline.py --dataset_name citeseer --method hgnn`

For all runtime options, please see either utils.py or run:

`python hypergraph.py --h`


Can run such as:

python hypergraph.py --dataset_name citeseer --seed
python hypergraph.py --dataset_name dblp --seed --n_layers 2
python hypergraph.py --dataset_name cora --seed --do_svd

#to test alpha/beta, adjust code for alpha range
python hypergraph.py --dataset_name citeseer --seed

#to predict edges
python hypergraph.py --dataset_name citeseer --seed --predict_edge
emacs hypergraph.py --seed --predict_edge --dataset_name citesser

#Experiment with edge linear:
python hypergraph.py --dataset_name cora --seed --edge_linear

python hypergraph.py --seed --predict_edge --dataset_name citeseer --n_layers 1

To run baselines:
python baselines.py
python baseline.py --predict_edge --dataset_name citeseer --method hgnn
python baseline.py --dataset_name pubmed --method hgnn
python baseline.py --dataset_name dblp --method hgnn

python baseline.py --dataset_name cora --do_svd --method hgnn
