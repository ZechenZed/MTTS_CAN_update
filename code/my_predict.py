import random
import argparse
import numpy as np
from math import log
from model import MTTS_CAN
from statistics import mean
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
from scipy.interpolate import interp1d
from inference_preprocess import preprocess_raw_video


def filter_fxn(pre_HR, cur_HR, cap):
    gap = pre_HR - cur_HR
    if abs(gap) < cap:
        return cur_HR
    else:
        return pre_HR - np.sign(gap) * cap * log(abs(gap) + 1)


def peaks():
    BP_test = np.loadtxt('C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Ground_truth/BP_raw_1KHz/F001-T7-BP.txt')
    size = int(BP_test.shape[0] // 40 * 40 / 40)
    print(size)
    BP_mean = np.zeros(size)
    for i in range(0, size):
        BP_mean[i] = mean(BP_test[i*40:(i +1)* 40])
    maxBP = max(BP_mean)
    minBP = min(BP_mean)
    threshold = 0.5 * (maxBP - minBP)
    print('max BP is', maxBP, 'min BP is', minBP)
    print('---> threshold/horizontal distance =', threshold)
    BP_systolic, _ = find_peaks(BP_mean, distance=10)
    BP_inter = np.zeros(BP_mean.shape[0])
    prev_index = 0
    for index in BP_systolic:
        y_interp = interp1d([prev_index,index],[BP_mean[prev_index], BP_mean[index]])
        for i in range(prev_index,index+1):
            BP_inter[i] = y_interp(i)
        prev_index = index

    plt.plot(BP_inter, label='interp')
    # plt.plot(BP_systolic, BP_mean[BP_systolic], label='Systolic Pressure Peaks')
    plt.plot(BP_mean, label='Mean')
    # plt.plot(BP_test, label='Original BP wave')
    plt.legend()
    plt.show()


def my_predict(data_type, dataset_type):
    img_rows = 48
    img_cols = 48
    frame_depth = 1
    path = 'C:/Users/Zed/Desktop/Project-BMFG'
    model_checkpoint = path + '/checkpoints/mtts_sys_kernal66_' + dataset_type + '_drop2.hdf5'
    batch_size = 32

    dXsub = np.load(path + '/preprocessed_v4v/' + data_type + '_frames_' + dataset_type + '.npy')
    BP_gt = np.load(path + '/preprocessed_v4v/' + data_type + '_BP_mean_systolic.npy')
    size = dXsub.shape[0]
    print('*************', size, '*************')
    # print(BP_gt.shape)
    x = random.randint(0, size - 25 * 500 - 1)
    dXsub = dXsub[x:x + 25 * 500]
    BP_gt = BP_gt[x:x + 25 * 500]

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)
    BP_pred = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    plt.plot(BP_pred, label='prediction')
    plt.plot(BP_gt, label='ground truth')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data_type', type=str, default='test',
                        help='data type')
    parser.add_argument('-dataset', '--dataset_type', type=str, default='face_large',
                        help='dataset type')
    args = parser.parse_args()
    my_predict(args.data_type, args.dataset_type)
    # preprocess_raw_video('C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Videos/train/F001_T1.mkv')
    # peaks()
