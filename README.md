## Required Packages

python 3.7
pytorch 1.11.0  Cuda 11.3
Deep Graph Library 0.8.0  Cuda 11.3
numpy 1.21.2
torch_scatter 2.0.9
networkx 2.6.3
pickle 
scikit-learn 1.0.2



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