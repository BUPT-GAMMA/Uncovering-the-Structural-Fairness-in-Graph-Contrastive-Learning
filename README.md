# GRADE

Source code for NeurIPS 2022 paper ["**Uncovering the Structural Fairness in Graph Contrastive Learning**"](https://arxiv.org/abs/2210.03011)

## Environment Settings

* python == 3.7
* torch == 1.11.0, cuda == 11.3
* Deep Graph Library == 0.8.0
* numpy == 1.21.2
* torch_scatter == 2.0.9
* networkx == 2.6.3
* scikit-learn == 1.0.2

## Main Parameter Settings

- model
  
  - threshold: the threshold to divide tail and head nodes
  - temp: the temperature for similarity
  - der1: the drop edge ratio of the 1st augmentation
  - der2: the drop edge ratio of the 2nd augmentation
  - dfr1: the drop feature ratio of the 1st augmentation
  - dfr2: the drop feature ratio of the 2nd augmentation

- trainer
  
  - mode: train-test split setting (full/part)
  - warmup: the warmup epochs of training

## Files in the folder

```
GRADE/
├── code/
│   ├── main.py: training the GRADE model
│   ├── aug.py: the implementation of the proposed degree-aware augmentation
│   ├── model.py
│   └── utils.py
├── data/
└── README.md
```

## Main Results

To replicate GRADE results from Table 1 and Table 2, run

```
# Cora dataset

python main.py --dataset cora --mode full --hid_dim 256 --out_dim 256 --act_fn relu --temp 0.5 --save_name best_cora.pkl --test

# Citeseer dataset

python main.py --dataset citeseer --mode full --hid_dim 256 --out_dim 256 --act_fn relu --temp 1.7 --save_name best_citeseer.pkl --test

# Photo dataset

python main.py --dataset photo --mode full --hid_dim 512 --out_dim 512 --act_fn relu --temp 0.8 --save_name best_photo.pkl --test

# Computer dataset

python main.py --dataset computer --mode full --hid_dim 800 --out_dim 800 --act_fn prelu --temp 1.1 --save_name best_computer.pkl --test
```

# Reference

```
@inproceedings{wang2022uncovering,
  title={Uncovering the Structural Fairness in Graph Contrastive Learning},
  author={Wang, Ruijia and Wang, Xiao and Shi, Chuan and Song, Le},
  booktitle={Proceedings of 36th Conference on Neural Information Processing Systems},
  year={2022}
}
```
