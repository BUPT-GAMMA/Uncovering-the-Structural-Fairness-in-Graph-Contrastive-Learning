from os import replace
import numpy as np

import torch as th
import torch.nn.functional as F
from torch_scatter import scatter_add
import dgl

from utils import get_sim



def random_aug(graph, x, feat_drop_rate, edge_mask_rate):
    n_node = graph.num_nodes()

    edge_mask = random_mask_edge(graph, edge_mask_rate)
    feat = drop_feature(x, feat_drop_rate)

    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()
    return ng, feat


def degree_aug(graph, x, embeds, degree, 
        feat_drop_rate_1, edge_mask_rate_1, feat_drop_rate_2, edge_mask_rate_2, 
        threshold):
    feat1 = drop_feature(x, feat_drop_rate_1)
    feat2 = drop_feature(x, feat_drop_rate_2)

    max_degree = np.max(degree)
    node_dist = get_node_dist(graph)
    src_idx = th.LongTensor(np.argwhere(degree < threshold).flatten()).to(x.device)
    rest_idx = th.LongTensor(np.argwhere(degree >= threshold).flatten()).to(x.device)
    rest_node_degree = degree[degree>=threshold]
    
    sim = get_sim(embeds, embeds)
    sim = th.clamp(sim, 0, 1)
    sim = sim - th.diag_embed(th.diag(sim))
    src_sim = sim[src_idx]
    # dst_idx = th.argmax(src_sim, dim=-1).to(x.device)
    dst_idx = th.multinomial(src_sim + 1e-12, 1).flatten().to(x.device)

    rest_node_degree = th.LongTensor(rest_node_degree)
    degree_dist = scatter_add(th.ones(rest_node_degree.size()), rest_node_degree).to(x.device)
    prob = degree_dist.unsqueeze(dim=0).repeat(src_idx.size(0), 1)
    # aug_degree = th.argmax(prob, dim=-1).to(x.device)
    aug_degree = th.multinomial(prob, 1).flatten().to(x.device)

    new_row_mix_1, new_col_mix_1 = neighbor_sampling(src_idx, dst_idx, node_dist, sim, 
                                                    max_degree, aug_degree)
    new_row_rest_1, new_col_rest_1 = degree_mask_edge(rest_idx, sim, max_degree, rest_node_degree, edge_mask_rate_1)
    nsrc1 = th.cat((new_row_mix_1, new_row_rest_1)).cpu()
    ndst1 = th.cat((new_col_mix_1, new_col_rest_1)).cpu()

    ng1 = dgl.graph((nsrc1, ndst1), num_nodes=graph.num_nodes()).to_simple().to(x.device)
    # ng1 = dgl.graph((nsrc1, ndst1), num_nodes=graph.num_nodes()).to(x.device)
    ng1 = ng1.add_self_loop()

    new_row_mix_2, new_col_mix_2 = neighbor_sampling(src_idx, dst_idx, node_dist, sim, 
                                                    max_degree, aug_degree)
    new_row_rest_2, new_col_rest_2 = degree_mask_edge(rest_idx, sim, max_degree, rest_node_degree, edge_mask_rate_2)
    nsrc2 = th.cat((new_row_mix_2, new_row_rest_2)).cpu()
    ndst2 = th.cat((new_col_mix_2, new_col_rest_2)).cpu()

    ng2 = dgl.graph((nsrc2, ndst2), num_nodes=graph.num_nodes()).to_simple().to(x.device)
    # ng2 = dgl.graph((nsrc2, ndst2), num_nodes=graph.num_nodes()).to(x.device)
    ng2 = ng2.add_self_loop()
    return ng1, ng2, feat1, feat2


def get_node_dist(graph):
    """
    Compute adjacent node distribution.
    """
    row, col = graph.edges()[0], graph.edges()[1]
    num_node = graph.num_nodes()

    dist_list = []
    for i in range(num_node):
        dist = th.zeros([num_node], dtype=th.float32, device=graph.device)
        idx = row[(col==i)]
        dist[idx] = 1
        dist_list.append(dist)
    dist_list = th.stack(dist_list, dim=0)
    return dist_list


def drop_feature(x, drop_prob):
    drop_mask = th.empty((x.size(1),),
                        dtype=th.float32,
                        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def neighbor_sampling(src_idx, dst_idx, node_dist, sim, 
                    max_degree, aug_degree):
    phi = sim[src_idx, dst_idx].unsqueeze(dim=1)
    phi = th.clamp(phi, 0, 0.5)

    # print('phi', phi)
    mix_dist = node_dist[dst_idx]*phi + node_dist[src_idx]*(1-phi)

    new_tgt = th.multinomial(mix_dist + 1e-12, int(max_degree)).to(phi.device)
    tgt_idx = th.arange(max_degree).unsqueeze(dim=0).to(phi.device)

    new_col = new_tgt[(tgt_idx - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = src_idx.repeat_interleave(aug_degree)
    return new_row, new_col
    

def degree_mask_edge(idx, sim, max_degree, node_degree, mask_prob):
    aug_degree = (node_degree * (1- mask_prob)).long().to(sim.device)
    sim_dist = sim[idx]

    # _, new_tgt = th.topk(sim_dist + 1e-12, int(max_degree))
    new_tgt = th.multinomial(sim_dist + 1e-12, int(max_degree))
    tgt_idx = th.arange(max_degree).unsqueeze(dim=0).to(sim.device)

    new_col = new_tgt[(tgt_idx - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = idx.repeat_interleave(aug_degree)
    return new_row, new_col

def random_mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = th.FloatTensor(np.ones(E) * mask_prob)
    masks = th.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx