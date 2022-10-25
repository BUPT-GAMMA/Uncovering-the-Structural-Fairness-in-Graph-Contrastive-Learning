import numpy as np
import functools
import networkx as nx
import pickle as pkl

import torch as th
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset



def get_mask(idx, l):
    """Create mask."""
    mask = th.zeros(l, dtype=th.bool)
    mask[idx] = 1
    return mask


def get_sim(embeds1, embeds2):
    # normalize embeddings across feature dimension
    embeds1 = F.normalize(embeds1)
    embeds2 = F.normalize(embeds2)
    sim = th.mm(embeds1, embeds2.t())
    return sim


def load(name, mode):
    assert name in ['cora', 'citeseer', 'photo', 'computer']
    assert mode in ['full', 'part']
    path = '../data/'
    if name == 'cora':
        dataset = CoraGraphDataset(path)
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset(path)
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset(path)
    elif name == 'computer':
        dataset = AmazonCoBuyComputerDataset(path)

    graph = dataset[0]

    feat = graph.ndata['feat']
    labels = graph.ndata['label']

    nxg = graph.to_networkx()
    adj = nx.to_scipy_sparse_matrix(nxg, dtype=np.float)
    degree = np.array(adj.sum(0)).squeeze()

    num_class = dataset.num_classes
    num_node = graph.num_nodes()

    idx_test = [i for i in range(num_node) if degree[i]>0 and degree[i]<50]
    idx_test = idx_test[:1000]

    if mode == 'full':
        idx_train = [i for i in range(num_node) if i not in idx_test]
    elif mode == 'part':
        idx_train = []
        for j in range(num_class):
            idx_train.extend([i for i,x in enumerate(labels) if x==j and i not in idx_test][:50])

    train_mask = get_mask(idx_train, num_node)
    test_mask = get_mask(idx_test, num_node)
    return graph, feat, labels, train_mask, test_mask, degree


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {'mean': np.mean(values), 'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


@repeat(1)
def linear_clf(embeddings, y, train_mask, test_mask, degree, dataset):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = Y[train_mask]
    y_test = Y[test_mask]
    degree = degree[test_mask]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    # with open('{}_embedding.p'.format(dataset), 'ab+') as f:
    #     pkl.dump((y_pred, y_test, degree), f)
    # f.close()

    y_pred = prob_to_one_hot(y_pred)

    acc = (np.argmax(y_test, axis=1) == np.argmax(y_pred, axis=1)).astype(float)
    degree_dict = {}
    for i in range(degree.shape[0]):
        if degree[i] not in degree_dict:
            degree_dict[degree[i]] = []
        degree_dict[degree[i]].append(acc[i])

    for d,l in degree_dict.items():
        degree_dict[d] = np.mean(l)
    bias = np.var(list(degree_dict.values()))
    mean = np.mean(list(degree_dict.values()))

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    return {'F1Mi': micro, 'F1Ma': macro, 'Mean':mean, 'Bias':bias}
