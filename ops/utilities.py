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


def resample_image_lists(im_list, lab_list, sp):
    # Shuffle each it_list then sample sp
    sampled_ims = []
    sampled_labs = []
    for im, il, itsp in zip(im_list, lab_list, sp):
        npim = np.asarray(im)
        npil = np.asarray(il)
        shuffle = np.random.permutation(len(im))
        sampled_ims += [list(npim[shuffle[:itsp]])]
        sampled_labs += [list(npil[shuffle[:itsp]])]
    return sampled_ims, sampled_labs

def shuffle_list(l):
    return list(np.asarray(l)[np.random.permutation(len(l))])
