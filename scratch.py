import numpy as np
import pickle
import torch

from utils.train_synthetic import run_synthetic_experiments, load_synthetic_results
from utils.results import *

for baseline in ['DPRNN', 'QRNN', 'CFRNN']:
    for seed in range(5):
        print(seed)
        run_synthetic_experiments(experiment='sample_complexity',
                                  baseline=baseline,
                                  retrain_auxiliary=True,
                                  save_model=True,
                                  save_results=True,
                                  rnn_mode='LSTM',
                                  seed=seed)

