
import yaml

import argparse

import numpy as np

from joblib import Parallel, delayed

from sklearn.metrics import mutual_info_score

def discretize_vector(v, bins):
    quantiles = [i/bins for i in range(1, bins)]
    bins = np.quantile(v, quantiles)
    v_ = np.digitize(v, bins=bins)
    return v_

def discretize_matrix(M, bins=4):
  return np.array(Parallel(n_jobs=-1)(delayed(discretize_vector)(M[:, i], bins) for i in range(M.shape[1]))).T

def MI_vector_matrix(v1, M2, bins):
  mi = []
  for i in range(M2.shape[1]):
    mi_score =  mutual_info_score(v1, M2[:, i])
    mi.append(mi_score)
  return mi

def MI_matrix_matrix(M1, M2, bins):
  mi = Parallel(n_jobs=-1)(delayed(MI_vector_matrix)(M1[:, i], M2, bins) for i in range(M1.shape[1]))
  return np.array(mi)

def pairs_to_graph(tokens, pairs):
    n = len(tokens)
    A = np.zeros((n, n))
    token_2_id = {token : i for i, token in enumerate(tokens)}
    for (a, b) in pairs:
        A[token_2_id[a], token_2_id[b]] = 1
    return A

class YamlNamespace(argparse.Namespace):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [YamlNamespace(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, YamlNamespace(b) if isinstance(b, dict) else b)

def _parse_args():
    parser = argparse.ArgumentParser("Training script")
    parser.add_argument("--config", "-c", type=str, required=True, help="The YAML config file")
    cli_args = parser.parse_args()
    with open(cli_args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config_ = YamlNamespace(config)
    return config_, config
