
import os
import sys
import yaml

import numpy as np
import pandas as pd
import datetime

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

sys.path.append(".")

from deepNF.mda import mda
from deepNF.preprocessing import load_graph, RWR, PPMI
from deepNF.utils import EarlyStopping, YamlNamespace, _parse_args


if __name__ == "__main__":

    config, config_yaml = _parse_args()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    results_folder = os.path.join(config.results_folder, str(datetime.datetime.now()))
    os.mkdir(results_folder)
    with open(os.path.join(results_folder, "config.yaml"), "w") as f:
        f.write(yaml.dump(config_yaml))

    networks_files = []
    for networks_folder in config.networks_folders:
        paths = list(os.listdir(networks_folder))
        networks_files += [os.path.join(networks_folder, path) for path in paths]
    date2tokens = dict()
    date2nets = dict()
    methods = set()

    for file in networks_files:
        meta = file.split("/")[-1].split("_")
        method = (meta[0], meta[1], meta[3]) # pour le MI : ('MI', 'lookback, d, bins', 'lag'), pour secteur ('SECTOR', '', '')
        date = meta[2]
        try:
            (tokens, A) = load_graph(file)
            if not(date in date2nets.keys()):
                date2nets[date] = dict()
                date2tokens[date] = tokens
            date2nets[date][method] = A
            methods.add(method)
        except:
            continue

    methods = list(methods)

    metadata_samples = []
    samples = []
    for date in date2nets.keys():
        tokens = date2tokens[date]
        adj_matrices = [date2nets[date][method] for method in methods]
        rwr_matrices = [RWR(A) for A in adj_matrices]
        ppmi_matrices = [PPMI(R) for R in rwr_matrices]
        for method, A in zip(methods, adj_matrices):
            if A.shape != (100, 100):
                print(method)
                print(date)
        for i, token in enumerate(tokens):
            metadata_samples.append((date, token))
            samples.append(np.array([P[i, :] for P in ppmi_matrices]))
    samples = np.array(samples)
    train_samples, val_samples = train_test_split(samples, test_size=0.2)
    train_samples = train_samples.astype(np.float32)
    val_samples = val_samples.astype(np.float32)
    train_samples, val_samples = torch.from_numpy(train_samples), torch.from_numpy(val_samples)
    train_dataset, val_dataset = TensorDataset(train_samples), TensorDataset(val_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)


    model = mda(N=len(methods), M=config.M, input_dims=len(methods) * [config.input_dim],
                dims=config.dims, activation=config.activation, dropout=config.dropout)

    optimizer = Adam(params=model.parameters(),
                     lr=config.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, threshold=1e-3)
    loss = nn.MSELoss()

    for epoch in range(config.num_epochs):

        train_running_loss = 0.
        val_running_loss = 0.

        model.train()
        for batch in train_dataloader:
            model.zero_grad()
            out = model(batch[0])
            mse_error = torch.zeros(1)
            for n in range(out.shape[1]):
                mse_error += loss(batch[0][:, n, :], out[:, n, :])
            mse_error /= out.shape[1]
            mse_error.backward()
            optimizer.step()
            train_running_loss += mse_error.item()
        train_running_loss /= len(train_dataloader)

        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                out = model(batch[0])
                mse_error = loss(batch[0], out)
                val_running_loss += mse_error.item()
            val_running_loss /= len(val_dataloader)
            scheduler.step(val_running_loss)

        if epoch % config.res_plot == 0:
            print("epoch_{0} | train_loss_{1} | val_loss_{2}".format(epoch, train_running_loss, val_running_loss))

    samples = torch.from_numpy(samples.astype(np.float32))
    dataset = TensorDataset(samples)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            emb = model.encode(batch[0])
            emb = emb.detach().numpy()
            embeddings.extend(emb)
    pd.DataFrame(data=embeddings, index=metadata_samples).to_csv(os.path.join(results_folder, "embeddings.csv"))
