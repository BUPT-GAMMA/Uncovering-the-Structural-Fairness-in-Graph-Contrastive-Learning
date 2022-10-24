import numpy as np
import argparse
import warnings

import torch as th
import torch.nn as nn

from model import GRADE
from aug import degree_aug, random_aug
from utils import load, linear_clf



warnings.filterwarnings('ignore')

def count_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--gpu_id', type=int, default=9)
parser.add_argument('--test', action='store_true', default=False, help='Train/Test mode.')
parser.add_argument('--mode', type=str, default='full')

parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay.')
parser.add_argument('--temp', type=float, default=1.0, help='Temperature.')
parser.add_argument('--threshold', type=int, default=9, help='Definition of low-degree nodes.')
parser.add_argument('--warmup', type=int, default=200, help='Warmup of training.')

parser.add_argument('--act_fn', type=str, default='relu')

parser.add_argument("--hid_dim", type=int, default=256, help='Hidden layer dim.')
parser.add_argument("--out_dim", type=int, default=256, help='Output layer dim.')

parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument('--der1', type=float, default=0.2, help='Drop edge ratio of the 1st augmentation.')
parser.add_argument('--der2', type=float, default=0.2, help='Drop edge ratio of the 2nd augmentation.')
parser.add_argument('--dfr1', type=float, default=0.2, help='Drop feature ratio of the 1st augmentation.')
parser.add_argument('--dfr2', type=float, default=0.2, help='Drop feature ratio of the 2nd augmentation.')
parser.add_argument('--save_name', type=str, default='try.pkl', help='save ckpt name')

args = parser.parse_args()

if args.gpu_id != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu_id)
else:
    args.device = 'cpu'

if __name__ == '__main__':
    lr = args.lr
    hid_dim = args.hid_dim
    out_dim = args.out_dim

    num_layers = args.num_layers
    act_fn = ({'relu': nn.ReLU(), 'prelu': nn.PReLU()})[args.act_fn]

    drop_edge_rate_1 = args.der1
    drop_edge_rate_2 = args.der2
    drop_feature_rate_1 = args.dfr1
    drop_feature_rate_2 = args.dfr2

    temp = args.temp
    epochs = args.epochs
    wd = args.wd

    graph, feat, labels, train_mask, test_mask, degree = load(args.dataset, args.mode)
    in_dim = feat.shape[1]

    model = GRADE(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
    model = model.to(args.device)
    print(f'# params: {count_parameters(model)}')

    if not args.test:
        optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        model.train()
        graph = graph.to(args.device)
        feat = feat.to(args.device)
        for epoch in range(epochs):
            if epoch < args.warmup:
                graph1, feat1 = random_aug(graph, feat, drop_feature_rate_1, drop_edge_rate_1)
                graph2, feat2 = random_aug(graph, feat, drop_feature_rate_2, drop_edge_rate_2)

            else:
                graph = graph.add_self_loop()
                embeds = model.get_embedding(graph, feat)

                graph = graph.remove_self_loop()
                graph1, graph2, feat1, feat2= degree_aug(graph, feat, embeds, degree, 
                                            drop_feature_rate_1, drop_edge_rate_1, drop_feature_rate_2, drop_edge_rate_2, 
                                            args.threshold)

            optimizer.zero_grad()
            loss = model(graph1, graph2, feat1, feat2)
            loss.backward()
            optimizer.step()
            print(f'Epoch={epoch:03d}, loss={loss.item():.4f}')

    print("=== Final ===")
    if args.test:
        model.load_state_dict(th.load(args.save_name, map_location=args.device))
        graph = graph.to(args.device)
        feat = feat.to(args.device)
    else:
        th.save(model.state_dict(), args.save_name)
    graph = graph.add_self_loop()
    embeds = model.get_embedding(graph, feat)

    '''Evaluation Embeddings '''
    linear_clf(embeds, labels, train_mask, test_mask, degree, args.dataset)