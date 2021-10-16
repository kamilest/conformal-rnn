from utils.train_synthetic import run_synthetic_experiments

for baseline in ['BJRNN']:
    for seed in range(1):
        run_synthetic_experiments(experiment='time_dependent',
                                  baseline=baseline,
                                  n_train = 2000,
                                  save_model=False,
                                  save_results=True,
                                  rnn_mode='LSTM',
                                  seed=seed)