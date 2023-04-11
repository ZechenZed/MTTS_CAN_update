import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
import argparse
from scipy.signal import find_peaks
import os
import csv
import pandas as pd

sys.path.append('../')
from model import Attention_mask, MTTS_CAN, TS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, detrend

import numpy as np
from scipy.signal import periodogram


def prpsd(BVP, FS, LL_PR, UL_PR):
    """
    Estimates a pulse rate from a BVP signal.
    Inputs:
        BVP     - A BVP timeseries.
        FS      - The sample rate of the BVP time series (Hz/fps).
        LL_PR   - The lower limit for pulse rate (bpm).
        UL_PR   - The upper limit for pulse rate (bpm).
        PlotTF  - Boolean to turn plotting results on or off.
    Outputs:
        PR      - The estimated PR in BPM.
    """

    Nyquist = FS / 2
    FResBPM = 0.5  # resolution (bpm) of bins in power spectrum used to determine PR and SNR
    N = int((60 * 2 * Nyquist) / FResBPM)

    # Construct Periodogram
    F, Pxx = periodogram(BVP, window=np.hamming(len(BVP)), nfft=N, fs=FS)
    FMask = (F >= (LL_PR / 60)) & (F <= (UL_PR / 60))

    # Calculate predicted HR:
    FRange = F[FMask]
    PRange = Pxx[FMask]
    MaxInd = np.argmax(Pxx[FMask], axis=0)
    PR_F = FRange[MaxInd]
    PR = PR_F * 60

    return PR


def predict_vitals(video_name):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = './mtts_can.hdf5'
    batch_size = 100
    fs = 25
    sample_data_path = " ../Phase1_data/Videos/train-001_of_002/" + video_name + ".mkv"

    dXsub = preprocess_raw_video(sample_data_path, dim=36)
    print('dXsub shape', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth) * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    pulse_pred = yptest[0]
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    # resp_pred = yptest[1]
    # resp_pred = detrend(np.cumsum(resp_pred), 100)
    # [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    # resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))

    # Read the ground truth file and using sliding window to calculate the difference
    HR_predicted = np.ones(dXsub_len)

    with open("../Phase1_data/Ground_truth/Physiology/" + video_name + ".txt") as f:
        contents = f.read()
        contents = contents.split(", ")
        start = 2
        end = start + dXsub_len
        print('Total number of continuous ground truth frame:', end - start)
        window_size = 1
        HR_gt = [float(contents[start])]
        for i in range(start + 1, end):
            if contents[i] == contents[i - 1]:
                window_size += 1
            else:
                HR_pred_curr = prpsd(pulse_pred[i - window_size:i], fs, 40, 180)
                HR_predicted[i - window_size - 2:i - 1] = HR_pred_curr
                window_size = 1
            if i == end - 1:
                window_size += 1
                HR_pred_curr = prpsd(pulse_pred[i - window_size:i], fs, 40, 180)
                HR_predicted[i - window_size - 2:i - 1] = HR_pred_curr
            HR_gt.append(float(contents[i]))
        HR_gt = np.array(HR_gt)

    MAE = sum(abs(HR_predicted - HR_gt)) / dXsub_len
    RMSE = np.sqrt(sum(abs(HR_predicted - HR_gt) ** 2) / dXsub_len)
    print("MAE: ", MAE)
    print("RMSE: ", RMSE)

    return MAE, RMSE
    ################## Plot ##################
    # plt.subplot(211)
    # plt.plot(pulse_pred)
    # plt.title('Pulse Prediction')
    # plt.subplot(212)
    # plt.plot(resp_pred)
    # plt.title('Resp Prediction')
    # plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--video_name', type=str, help='processed video name')
    # parser.add_argument('--sampling_rate', type=int, default=25, help='sampling rate of your video')
    # parser.add_argument('--batch_size', type=int, default=100, help='batch size (multiplier of 10)')
    # args = parser.parse_args()

    dir_path = "../Phase1_data/Videos/train-001_of_002"
    res = []

    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)
    print(res)
    num_video = len(res)
    MAE_array = np.empty(num_video)
    RMSE_array = np.empty(num_video)

    for i in range(num_video):
        print("Current Video:", res[i])
        video_name = res[i][0:-4]
        MAE, RMSE = predict_vitals(video_name)
        MAE_array[i] = MAE
        RMSE_array[i] = RMSE
    print("Average MAE for 001:", sum(MAE_array) / num_video)
    print("Average RMSE for 001:", sum(RMSE_array) / num_video)
    np.savetxt("MAE_001", MAE_array, delimiter=" ")
    np.savetxt("RMSE_001", RMSE_array, delimiter=" ")
