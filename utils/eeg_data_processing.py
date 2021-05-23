import gzip
import os
import pathlib

eeg_root_train = 'data/eeg/SMNI_CMI_TRAIN'
eeg_root_test = 'data/eeg/SMNI_CMI_TEST'


def parse_eeg_file(filename):
    with gzip.open(filename, 'rb') as f:
        chans = {}
        for line in f:
            tokens = line.decode('ascii').split()
            if tokens[0] == '25':
                if tokens[1] not in chans.keys():
                    chans[tokens[1]] = []
                chans[tokens[1]].append(float(tokens[3]))
        chan_arrays = []
        for chan in chans.values():
            chan_arrays.append(chan)
    return chan_arrays


def parse_eeg_dataset(root):
    dataset = []
    for folder in os.listdir(root):
        if folder != 'README':
            subfolder = os.path.join(root, folder)
            for filename in os.listdir(subfolder):
                f = os.path.join(subfolder, filename)
                if '.gz' in pathlib.Path(f).suffixes:
                    chan_arrays = parse_eeg_file(f)
                    dataset.extend(chan_arrays)
    return dataset
