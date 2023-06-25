import argparse
from math import log
from model import MTTS_CAN
import matplotlib.pyplot as plt
import numpy as np
from inference_preprocess import preprocess_raw_video


def filter_fxn(pre_HR, cur_HR, cap):
    gap = pre_HR - cur_HR
    if abs(gap) < cap:
        return cur_HR
    else:
        return pre_HR - np.sign(gap) * cap * log(abs(gap) + 1)


def my_predict(data_type):
    img_rows = 48
    img_cols = 48
    frame_depth = 1
    model_checkpoint = 'C:/Users/Zed/Desktop/Project-BMFG/BMFG/checkpoints/mtts_v1_face_Adam.hdf5'
    batch_size = 32

    dXsub = np.load('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/' + data_type + '_frames_face.npy')
    BP_gt = np.load('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/' + data_type + '_BP_mean.npy')
    print(dXsub.shape)
    # print(BP_gt.shape)
    dXsub = dXsub[108360:108360+25*30]
    BP_gt = BP_gt[108360:108360+25*30]

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
    args = parser.parse_args()
    my_predict(args.data_type)
    # preprocess_raw_video('C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Videos/train/F001_T1.mkv')
