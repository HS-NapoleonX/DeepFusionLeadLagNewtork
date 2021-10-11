
import os
import sys
import pickle

import numpy as np
import pandas as pd

from scipy.stats import gamma
import datetime, pytz

from tqdm import tqdm

sys.path.append(".")

from lead_lag_networks.utils import discretize_matrix, MI_matrix_matrix, pairs_to_graph
from lead_lag_networks.utils import YamlNamespace, _parse_args

if __name__ == "__main__":

    config, _ = _parse_args()

    results_folder = os.path.join(config.results_folder, str(datetime.datetime.now()))
    os.mkdir(results_folder)

    EPSILON = 1E-9
    ALPHA = 1 / 2 * (config.bins - 1) ** 2
    BETA = 1 / (config.lookback * np.log(2))

    cmc = pd.read_csv(config.ranking_file, index_col=0)
    cmc.index = pd.to_datetime(cmc.index)
    start_date = datetime.datetime.strptime(config.start_date, "%Y-%m-%d")
    cmc = cmc.loc[cmc.index >= start_date]

    dates = np.array(cmc.index.values)
    indices = [i for i in range(len(dates)) if i % config.sampling_period == (len(dates)-1) % config.sampling_period]
    dates = dates[indices]

    def build_lead_lag_network(df, date, lag):
        """
        df : matrix of prices
        date : date for which compute the network
        lag : time lag between two series
        """
        dates_ = np.array(df.loc[df.index <= date].index)
        indices_ = [i for i in range(len(dates_)) if i % config.d == (len(dates_) - 1) % config.d]
        dates_ = dates_[indices_]
        sub_df = df.loc[dates_].tail(config.lookback + 1 + lag).dropna(axis=1).iloc[:, :100]
        s_threshold = gamma.ppf(1 - 0.01 / (len(sub_df.columns) ** 2), a=ALPHA, scale=BETA)

        returns = np.log((sub_df + EPSILON) / (sub_df.shift(1) + EPSILON))
        returns = returns.iloc[1:, :].fillna(0).replace(np.inf, 0)
        returns = pd.DataFrame(data=discretize_matrix(returns.values, bins=config.bins), index=returns.index, columns=returns.columns)
        A = returns.head(config.lookback).values
        B = returns.tail(config.lookback).values

        C = np.array(MI_matrix_matrix(A, B, bins=config.bins)).T
        tokens = returns.columns
        pairs = [(tokens[i], tokens[j]) for i in range(len(tokens)) for j in range(len(tokens)) if C[i, j] >= s_threshold]

        return tokens, pairs

    res = dict()

    for current_date in dates:
        timestamp = pd.Timestamp(current_date, tz=pytz.utc)
        prices = pd.DataFrame()
        tokens = list(cmc.loc[current_date].values)
        for token in tokens:
            price = pd.read_csv(os.path.join(config.prices_folder, "{0}.csv".format(token)), index_col=0)
            price = price.loc[~price.index.duplicated(keep='first')]
            price.index = pd.to_datetime(price.index)
            prices[token] = price.open
        for lag in range(config.max_lag):
            tokens, pairs = build_lead_lag_network(prices, timestamp, lag)
            if lag <= config.max_lag_to_save:
                A = pairs_to_graph(tokens, pairs)
                with open(os.path.join(results_folder, "MI_{0},{1},{2}_{3}_{4}.pkl".format(config.lookback, config.d, config.bins, timestamp.date(), lag)), "wb") as pkl_file:
                    pickle.dump((tokens, A), pkl_file)
            res[(current_date, lag)] = pairs

    with open(os.path.join(results_folder, "pairs_{0}_{1}_{2}.pkl".format(config.lookback, config.d, config.bins)), "wb") as pkl_file:
        pickle.dump(res, pkl_file)
