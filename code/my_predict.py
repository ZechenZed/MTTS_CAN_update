import os
import random
import argparse
import numpy as np
from math import log
from model import MTTS_CAN
from statistics import mean
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
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


def my_predict(data_type, dataset_type, kernal):
    img_rows = 48
    img_cols = 48
    frame_depth = 1
    batch_size = 32
    path = 'C:/Users/Zed/Desktop/Project-BMFG'
    model_checkpoint = path + '/checkpoints/mtts_sys_kernal' + kernal + '_' + dataset_type + '_drop2_nb256.hdf5'

    # model_checkpoint = glob_path + '/checkpoints/mtts_sys_kernal' + kernal + '_' + dataset_type + '_drop2_nb256.hdf5'
    # glob_path = 'C:/Users/Zed/Desktop/Project-BMFG'
    # # video_folder_path = str()
    # if data_type != 'test':
    #     video_folder_path = glob_path + '/Phase1_data/Videos/' + data_type + '/'
    # else:
    #     video_folder_path = glob_path + '/Phase2_data/Videos/' + data_type + '/'
    #
    # video_path = []
    # for path in sorted(os.listdir(video_folder_path)):
    #     if os.path.isfile(os.path.join(video_folder_path, path)):
    #         video_path.append(path)
    # sample_video_name = video_path[random.randint(0, len(video_path))][0:-4]
    # sample_video_path = video_folder_path + sample_video_name + '.mkv'
    # dXsub = preprocess_raw_video(sample_video_path, dim=48)
    # dXsub_len = (dXsub.shape[0] // frame_depth) * frame_depth
    # dXsub = dXsub[:dXsub_len, :, :, :]
    #
    # BP_folder_path = str()
    # if data_type != 'train':
    #     BP_folder_path = glob_path + '/Phase2_data/blood_pressure/' + data_type + '_set_bp/'
    # else:
    #     BP_folder_path = glob_path + '/Phase1_data/Ground_truth/BP_raw_1KHz/'
    #
    # BP_sample_name = sample_video_name[0:4] + '-' + sample_video_name[5:8]
    # print(BP_sample_name)
    # if '.' in BP_sample_name:
    #     BP_sample_name += 'txt'
    # else:
    #     BP_sample_name += '.txt'
    #
    # sample_BP_file = np.loadtxt(BP_folder_path + BP_sample_name)
    # size = sample_BP_file.shape[0]
    # BP_systolic, _ = find_peaks(sample_BP_file, distance=10)
    # BP_inter = np.zeros(sample_BP_file.shape[0])
    # prev_index = 0
    # for index in BP_systolic:
    #     y_interp = interp1d([prev_index, index], [sample_BP_file[prev_index], sample_BP_file[index]])
    #     for i in range(prev_index, index + 1):
    #         BP_inter[i] = y_interp(i)
    #     prev_index = index
    # y_interp = interp1d([prev_index, size - 1], [sample_BP_file[prev_index], sample_BP_file[size - 1]])
    # for i in range(prev_index, size):
    #     BP_inter[i] = y_interp(i)
    #
    # BP_gt = BP_inter[0:dXsub_len]

    dXsub = np.load(path + '/preprocessed_v4v/' + data_type + '_frames_' + dataset_type + '.npy')
    BP_gt = np.load(path + '/preprocessed_v4v/' + data_type + '_BP_mean_systolic.npy')
    size = dXsub.shape[0]
    print('*************', size, '*************')
    print(BP_gt.shape)
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
    parser.add_argument('-kernal', '--kernal_size', type=str, default='99',
                        help='dataset type')
    parser.add_argument('-dense', '--nv_dense', type=str, default='256',
                        help='dataset type')
    args = parser.parse_args()
    my_predict(args.data_type, args.dataset_type, args.kernal_size)
    # preprocess_raw_video('C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Videos/train/F001_T1.mkv')
    # peaks()
