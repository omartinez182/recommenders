# Benchmarking of Sequence & Session-Based Recommenders
 

This repository was forked from the [Microsoft Recommenders](https://github.com/microsoft/recommenders) repo. Our goal was to provide a script to run experiments and provide some benchmarks for different recommendation algorithms, mainly we were interested in benchmarking the [SLi-Rec](https://www.microsoft.com/en-us/research/uploads/prod/2019/07/IJCAI19-ready_v1.pdf), [GRU4Rec](https://arxiv.org/pdf/1511.06939.pdf), and the [SUM](https://arxiv.org/pdf/2102.09211.pdf) recommenders.


## Getting Started

1) To run the code locally, first clone this repository and install the requirements.

```bash
git clone https://github.com/omartinez182/recommenders.git && cd recommenders
pip install -r requirements.txt
```
2) Then you can proceed to run the experiment passing the datasets as arguments.

```bash
$ python3 experiments.py --dataset 'reviews_Movies_and_TV_5.json' --metadataset 'meta_Movies_and_TV.json'
```
You can replace the dataset for any of the following in this [link](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/) (those that have a timestamp for each interaction, or the history sequence).

## Citations
```
@repo{Recommenders,
  title={Recommenders},
  author={Microsoft},
  year={2022},
  repository={https://github.com/microsoft/recommenders}
}
```
