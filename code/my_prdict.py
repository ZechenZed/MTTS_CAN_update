import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
import argparse
from scipy.signal import find_peaks
import os
import csv
from scipy.stats import pearsonr
import json
from math import log

# from numba import cuda, jit , cuda, uint32, f8, uint8

sys.path.append('../')
from model import Attention_mask, MTTS_CAN, TS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from inference_preprocess import preprocess_raw_video, detrend

import numpy as np
from scipy.signal import periodogram
from joblib import Parallel, delayed
from statistics import mean


def filter_fxn(pre_HR, cur_HR, cap):
    gap = pre_HR - cur_HR
    if abs(gap) < cap:
        return cur_HR
    else:
        return pre_HR - np.sign(gap) * cap * log(abs(gap) + 1)


def my_predict(data_type):
    img_rows = 36
    img_cols = 36
    frame_depth = 1
    model_checkpoint = 'C:/Users/Zed/Desktop/Project-BMFG/BMFG/checkpoints/my_mtts_v3_best_nbdense_128.hdf5'
    batch_size = 32

    dXsub = np.load('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/' + data_type + '_frames_0.npy')
    BP_gt = np.load('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/' + data_type + '_BP_0.npy')

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)
    BP_pred = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    plt.plot(BP_pred[0:1000], label='prediction')
    plt.plot(BP_gt[0:1000], label='ground truth')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data_type', type=str, default='train',
                        help='data type')
    args = parser.parse_args()
    my_predict(args.data_type)
