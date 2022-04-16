from operator import add
from pprint import pprint

import pandas as pd
import numpy as np
import os
from collections import Counter, defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import savgol_filter, argrelextrema


from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

pd.set_option('display.width', None)
EXT = '.txt'
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-ngpu', default=1, type=int)

parser.add_argument('-lr', default=0.0001, type=float)
parser.add_argument('-hidden_dim', default=128, type=int)
parser.add_argument('-cross_val', default=True, type=bool)
parser.add_argument('-dl_n_epochs', default=1000, type=int)
parser.add_argument('-batch_size', default=100, type=int)
parser.add_argument('-cnn_threshold', default=0.6, type=float)
parser.add_argument('-smooth', default=False, type=bool)
parser.add_argument('-remove_sudden_peaks', default=False, type=bool)
parser.add_argument('-back_sub', default=False, type=bool)

parser.add_argument('-virus_type', default='Entero', type=str)
parser.add_argument('-n_classes', default=5, type=int)
parser.add_argument('-aug', default='',
                    type=str)  # data augmentation: ros / smote / add_noise / drift / quan / dropout / pool
parser.add_argument('-task', default='classify', type=str)

opt = parser.parse_args()

RESULT_PATH = '/data/karenyyy/Virus2022/Accurate_Virus_Identification'
FEATURESET_FILE = os.path.join(os.path.join(RESULT_PATH, 'dataset'),
                               f'{opt.virus_type}_dataset.csv')

if opt.virus_type == 'Avian':
    CLASSES = ['coronavirus', 'influenzaA', 'REO']
    CLASSES2IDX = {'coronavirus': 0, 'influenzaA': 1, 'REO': 2}


elif opt.virus_type == 'Entero':
    CLASSES = ['CVB1', 'CVB3', 'EV70', 'EV71', 'PV2']
    CLASSES2IDX = {'CVB1': 0, 'CVB3': 1, 'EV70': 2, 'EV71': 3, 'PV2': 4}


elif opt.virus_type == 'Resp':
    CLASSES = ['Human_FLUA', 'RSV', 'Rhino', 'FLUB']
    CLASSES2IDX = {'Human_FLUA': 0, 'FLUB': 1, 'Rhino': 2, 'RSV': 3}



elif opt.virus_type == 'influenza':
    CLASSES = ['H1N1', 'H3N2', 'H5N2', 'H7N2']
    CLASSES2IDX = {'H1N1': 0, 'H3N2': 1, 'H5N2': 2, 'H7N2': 3}



elif opt.virus_type == 'envelope':
    CLASSES = ['fluA', 'fluAB', 'human']
    CLASSES2IDX = {'CVB1': 0,
                   'CVB3': 1,
                   'EB70': 2,
                   'EV71': 3,
                   'PV2': 4,
                   'REO': 5,
                   'Rhino': 6}


