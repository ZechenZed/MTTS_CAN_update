import os
import random
import argparse
import numpy as np
from math import log
from model import MTTS_CAN, MT_CAN_3D
from statistics import mean
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from inference_preprocess import preprocess_raw_video
from my_train import new_data_process


def filter_fxn(pre_HR, cur_HR, cap):
    gap = pre_HR - cur_HR
    if abs(gap) < cap:
        return cur_HR
    else:
        return pre_HR - np.sign(gap) * cap * log(abs(gap) + 1)


def peaks():
    BP_test = np.loadtxt('C:/Users/Zed/Desktop/V4V/Phase1_data/Ground_truth/BP_raw_1KHz/F001-T7-BP.txt')
    size = int(BP_test.shape[0] // 40 * 40 / 40)
    print(size)
    BP_mean = np.zeros(size)
    for i in range(0, size):
        BP_mean[i] = mean(BP_test[i * 40:(i + 1) * 40])
    maxBP = max(BP_mean)
    minBP = min(BP_mean)
    threshold = 0.5 * (maxBP - minBP)
    print('max BP is', maxBP, 'min BP is', minBP)
    print('---> threshold/horizontal distance =', threshold)
    BP_systolic, _ = find_peaks(BP_mean, distance=10)
    BP_inter = np.zeros(BP_mean.shape[0])
    prev_index = 0
    for index in BP_systolic:
        y_interp = interp1d([prev_index, index], [BP_mean[prev_index], BP_mean[index]])
        for i in range(prev_index, index + 1):
            BP_inter[i] = y_interp(i)
        prev_index = index

    plt.plot(BP_inter, label='interp')
    # plt.plot(BP_systolic, BP_mean[BP_systolic], label='Systolic Pressure Peaks')
    plt.plot(BP_mean, label='Mean')
    # plt.plot(BP_test, label='Original BP wave')
    plt.legend()
    plt.show()


def my_predict(data_type, dataset_type, kernal, dim=36):
    img_rows = dim
    img_cols = dim
    frame_depth = 5200
    batch_size = 6
    path = 'C:/Users/Zed/Desktop/Project-BMFG'
    # model_checkpoint = path + '/checkpoints/mtts_sys_kernal' + kernal + '_' + dataset_type + '_drop2_nb256.hdf5'
    model_checkpoint = path + '/checkpoints/mt3d_sys_face_large.hdf5'

    video = preprocess_raw_video('C:/Users/Zed/Desktop/V4V/Phase1_data/Videos/train/F001_T1.mkv')

    # video = video.reshape((-1, video.shape[0], video.shape[1], video.shape[2], video.shape[3]))
    video = np.reshape(video, (-1, video.shape[1], video.shape[2], video.shape[0], video.shape[3]))
    dXsub = np.zeros((1, 36, 36, 5200, 6))
    dXsub[0, :, :, 0:video.shape[3], :] = video
    print(dXsub.shape)

    model = MT_CAN_3D(frame_depth, 32, 64, (img_rows, img_cols, frame_depth, 3))
    model.load_weights(model_checkpoint)
    BP_pred = model.predict((dXsub[:, :, :, :, :3], dXsub[:, :, :, :, -3:]), batch_size=batch_size, verbose=1)
    print(BP_pred.shape)
    BP_pred = BP_pred.reshape(-1, 1)
    plt.plot(BP_pred, label='prediction')
    # plt.plot(BP_gt, label='ground truth')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data_type', type=str, default='test',
                        help='data type')
    parser.add_argument('-dataset', '--dataset_type', type=str, default='face_large',
                        help='dataset type')
    parser.add_argument('-kernal', '--kernal_size', type=str, default='99',
                        help='dataset type')
    parser.add_argument('-dense', '--nv_dense', type=str, default='256',
                        help='dataset type')
    args = parser.parse_args()
    my_predict(args.data_type, args.dataset_type, args.kernal_size)
    # preprocess_raw_video('C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Videos/train/F001_T1.mkv')
    # peaks()