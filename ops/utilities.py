import os
import csv
import numpy as np
from glob import glob


def get_files(pointer):
    return glob(pointer)


def get_label_list(file):
    labels = []
    with open(file, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        [labels.append(row) for row in reader]
    return flatten_list(labels)


def flatten_list(li):
    return [item for sublist in li for item in sublist]


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def resample_image_lists(it_list, sp):
    # Shuffle each it_list then sample sp
    sampled_list = []
    for il, itsp in zip(it_list, sp):
        npil = np.asarray(il)[np.random.permutation(len(il))]
        sampled_list += [list(npil[:itsp])]
    return sampled_list
