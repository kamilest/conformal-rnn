import gzip
import os
import pathlib
import pickle

eeg_root_train = 'data/eeg/SMNI_CMI_TRAIN'
eeg_root_test = 'data/eeg/SMNI_CMI_TEST'


def parse_eeg_file(filename):
    with gzip.open(filename, 'rb') as f:
        chans = {}
        for line in f:
            tokens = line.decode('ascii').split()
            if tokens[0] != '#':
                if tokens[1] not in chans.keys():
                    chans[tokens[1]] = []
                chans[tokens[1]].append(float(tokens[3]))
        chan_arrays = []
        for chan in chans.values():
            chan_arrays.append(chan)
    return chan_arrays


def get_raw_eeg_data(split='train', include_alcoholic_class=False,
                     cached=True):
    if split == 'train':
        root = eeg_root_train
    else:
        root = eeg_root_test

    if cached:
        with open('data/eeg_{}.pkl'.format(split), 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = []
        for folder in os.listdir(root):
            if folder != 'README' and (include_alcoholic_class or folder[3] == 'c'):
                subfolder = os.path.join(root, folder)
                for filename in os.listdir(subfolder):
                    f = os.path.join(subfolder, filename)
                    if '.gz' in pathlib.Path(f).suffixes:
                        chan_arrays = parse_eeg_file(f)
                        dataset.extend(chan_arrays)
        with open('data/eeg_{}.pkl'.format(split), 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset

# def get_eeg_splits(length=40, horizon=10, train=):

