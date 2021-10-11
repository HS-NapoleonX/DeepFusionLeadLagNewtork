
import yaml
import torch
import numpy as np

import argparse

class EarlyStopping:

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', save=True):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.save = save

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        if self.save:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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
