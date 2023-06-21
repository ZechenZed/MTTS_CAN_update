import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import json
import argparse
import numpy as np
from statistics import mean
from joblib import Parallel, delayed
from scipy.signal import medfilt

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from inference_preprocess import preprocess_raw_video, count_frames
from model import MTTS_CAN


# BP --> 25 Hz
def data_processing_1(data_type, device_type, dim=48):
    if device_type == "local":
        video_train_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Videos/train/"
        video_valid_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Videos/valid/"
        video_test_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase2_data/Videos/test/"
        BP_phase1_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Ground_truth/BP_raw_1KHz/"
        BP_val_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase2_data/blood_pressure/val_set_bp/"
        BP_test_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase2_data/blood_pressure/test_set_bp/"
    else:
        video_train_path = "/edrive2/zechenzh/V4V/Phase1_data/Videos/train/"
        video_valid_path = "/edrive2/zechenzh/V4V/Phase1_data/Videos/valid/"
        video_test_path = "/edrive2/zechenzh/V4V/Phase2_data/Videos/test/"
        BP_phase1_path = "/edrive2/zechenzh/V4V/Phase1_data/Ground_truth/BP_raw_1KHz/"
        BP_val_path = "/edrive2/zechenzh/V4V/Phase2_data/blood_pressure/val_set_bp/"
        BP_test_path = "/edrive2/zechenzh/V4V/Phase2_data/blood_pressure/test_set_bp/"

    video_folder_path = ""
    BP_folder_path = ""
    if data_type == "train":
        video_folder_path = video_train_path
        BP_folder_path = BP_phase1_path
    elif data_type == "test":
        video_folder_path = video_test_path
        BP_folder_path = BP_test_path
    else:
        video_folder_path = video_valid_path
        BP_folder_path = BP_val_path

    # Video path reading
    video_file_path = []
    for path in sorted(os.listdir(video_folder_path)):
        if os.path.isfile(os.path.join(video_folder_path, path)):
            video_file_path.append(path)
    num_video = len(video_file_path)
    print(num_video)

    videos = [Parallel(n_jobs=24*4)(
        delayed(preprocess_raw_video)(video_folder_path + video) for video in video_file_path)]
    videos = videos[0]

    tt_frame = 0
    for i in range(num_video):
        tt_frame += videos[i].shape[0] // 10 * 10

    # BP path reading
    BP_file_path = []
    for path in sorted(os.listdir(BP_folder_path)):
        if os.path.isfile(os.path.join(BP_folder_path, path)):
            BP_file_path.append(path)

    # BP & Video frame processing
    frames = np.zeros(shape=(tt_frame, dim, dim, 6))
    BP_lf = np.zeros(shape=tt_frame)
    frame_ind = 0
    for j in range(num_video):
        temp = np.loadtxt(BP_folder_path + BP_file_path[j])
        cur_frames = videos[j].shape[0] // 10 * 10
        temp_lf = np.zeros(cur_frames)
        frames[frame_ind:frame_ind + cur_frames, :, :, :] = videos[j][0:cur_frames, :, :, :]
        for i in range(0, cur_frames):
            temp_lf[i] = mean(temp[i * 40:(i + 1) * 40])
        BP_lf[frame_ind:frame_ind + cur_frames] = temp_lf
        frame_ind += cur_frames

    # Saving processed frames
    if device_type == "remote":
        np.save('/edrive2/zechenzh/preprocessed_v4v/' + data_type + '_frames_face.npy', frames)
        np.save('/edrive2/zechenzh/preprocessed_v4v/' + data_type + '_BP_mean.npy', BP_lf)
    else:
        np.save('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/' + data_type + '_frames_face.npy', frames)
        np.save('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/' + data_type + '_BP_mean.npy', BP_lf)


# Video --> 1000Hz
def data_processing_2(data_type, device_type, task_num):
    if device_type == "local":
        video_train_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Videos/train/"
        video_valid_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Videos/valid/"
        video_test_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase2_data/Videos/test/"
        BP_phase1_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Ground_truth/BP_raw_1KHz/"
        BP_test_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase2_data/blood_pressure/test_set_bp/"
    else:
        video_train_path = "../../../../edrive2/zechenzh/V4V/Phase1_data/Videos/train/"
        video_valid_path = "../../../../edrive2/zechenzh/V4V/Phase1_data/Videos/valid/"
        video_test_path = "../../../../edrive2/zechenzh/V4V/Phase2_data/Videos/test/"
        BP_phase1_path = "../../../../edrive2/zechenzh/V4V/Phase1_data/Ground_truth/BP_raw_1KHz/"
        BP_test_path = "../../../../edrive2/zechenzh/V4V/Phase2_data/blood_pressure/test_set_bp/"

    video_folder_path = ""
    BP_folder_path = ""
    if data_type == "train":
        video_folder_path = video_train_path
        BP_folder_path = BP_phase1_path
    else:
        video_folder_path = video_test_path
        BP_folder_path = BP_test_path

    # Video path reading
    video_file_path = []
    for path in sorted(os.listdir(video_folder_path)):
        if os.path.isfile(os.path.join(video_folder_path, path)):
            video_file_path.append(path)
    video_file_path = video_file_path[0:10]
    num_video = len(video_file_path)

    # Video processing
    videos = [Parallel(n_jobs=-1)(
        delayed(preprocess_raw_video)(video_folder_path + video) for video in video_file_path)]
    videos = videos[0]

    # BP path reading
    BP_file_path = []
    for path in sorted(os.listdir(BP_folder_path)):
        if os.path.isfile(os.path.join(BP_folder_path, path)):
            BP_file_path.append(path)

    # BP file reading
    BP_lf = []
    tt_frame = 0
    frame_video = []
    for i in range(num_video):
        BP_temp = np.loadtxt(BP_folder_path + BP_file_path[i])
        cur_frames = BP_temp.shape[0] // 1000 * 1000
        frame_video.append(cur_frames)
        tt_frame += cur_frames
        BP_temp_med = medfilt(BP_temp[0:cur_frames])
        for ind, element in enumerate(BP_temp_med):
            BP_lf.append(element)
    BP_lf = np.array(BP_lf)

    # BP & Video frame processing
    frames = np.zeros(shape=(tt_frame, 36, 36, 6))
    frame_ind = 0
    for i in range(num_video):
        temp_video = videos[i]
        cur_frames = frame_video[i]
        temp_video_expand = np.zeros(shape=(cur_frames, 36, 36, 6))
        for j in range(0, int(cur_frames / 40)):
            temp_video_expand[40 * j:40 * (j + 1), :, :, :] = temp_video[j, :, :, :]
        frames[frame_ind:frame_ind + cur_frames] = temp_video_expand
        frame_ind += cur_frames

    # Saving processed frames
    if device_type == "remote":
        np.save('../../../../edrive2/zechenzh/preprocessed_v4v/' + data_type + '_frames_v2.npy', frames)
        np.save('../../../../edrive2/zechenzh/preprocessed_v4v/' + data_type + '_BP_v2.npy', BP_lf)
    else:
        np.save('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/' + data_type + '_frames_v2.npy', frames)
        np.save('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/' + data_type + '_BP_v2.npy', BP_lf)
    print("###########Preprocess finished###########")


def data_processing_3(data_type, device_type, dim=48):
    if device_type == "local":
        video_train_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Videos/train/"
        video_valid_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Videos/valid/"
        video_test_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase2_data/Videos/test/"
        BP_phase1_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Ground_truth/BP_raw_1KHz/"
        BP_val_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase2_data/blood_pressure/val_set_bp/"
        BP_test_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase2_data/blood_pressure/test_set_bp/"
    else:
        video_train_path = "/edrive2/zechenzh/V4V/Phase1_data/Videos/train/"
        video_valid_path = "/edrive2/zechenzh/V4V/Phase1_data/Videos/valid/"
        video_test_path = "/edrive2/zechenzh/V4V/Phase2_data/Videos/test/"
        BP_phase1_path = "/edrive2/zechenzh/V4V/Phase1_data/Ground_truth/BP_raw_1KHz/"
        BP_val_path = "/edrive2/zechenzh/V4V/Phase2_data/blood_pressure/val_set_bp/"
        BP_test_path = "/edrive2/zechenzh/V4V/Phase2_data/blood_pressure/test_set_bp/"

    video_folder_path = ""
    BP_folder_path = ""
    if data_type == "train":
        video_folder_path = video_train_path
        BP_folder_path = BP_phase1_path
    elif data_type == "test":
        video_folder_path = video_test_path
        BP_folder_path = BP_test_path
    else:
        video_folder_path = video_valid_path
        BP_folder_path = BP_val_path

    # Video path reading
    video_file_path = []
    for path in sorted(os.listdir(video_folder_path)):
        if os.path.isfile(os.path.join(video_folder_path, path)):
            video_file_path.append(path)
    video_file_path = video_file_path[0:2]
    num_video = len(video_file_path)
    print(num_video)

    videos = [Parallel(n_jobs=10)(
        delayed(count_frames)(video_folder_path + video) for video in video_file_path)]
    videos = videos[0]

    tt_frame = 0
    for i in range(num_video):
        tt_frame += videos[i] // 10 * 10

    # BP path reading
    BP_file_path = []
    for path in sorted(os.listdir(BP_folder_path)):
        if os.path.isfile(os.path.join(BP_folder_path, path)):
            BP_file_path.append(path)

    # BP & Video frame processing
    BP_v3 = np.zeros(shape=(tt_frame, 40))
    frame_ind = 0
    for j in range(num_video):
        temp = np.loadtxt(BP_folder_path + BP_file_path[j])
        cur_frames = videos[j] // 10 * 10
        temp_BP = np.zeros(shape=(cur_frames, 40))
        for i in range(0, cur_frames):
            temp_BP[i, :] = temp[40 * i:40 * (i + 1)]
        BP_v3[frame_ind:frame_ind + cur_frames, :] = temp_BP
        frame_ind += cur_frames

    # Saving processed frames
    if device_type == "remote":
        np.save('/edrive2/zechenzh/preprocessed_v4v/' + data_type + '_BP_batch.npy', BP_v3)
    else:
        np.save('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/' + data_type + '_BP_batch.npy', BP_v3)


def model_train(data_type, device_type, task_num, nb_filters1, nb_filters2,
                dropout_rate1, dropout_rate2, nb_dense, nb_batch, nb_epoch, multiprocess):
    path = ""
    if device_type == "local":
        path = 'C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/'
    else:
        path = '/edrive2/zechenzh/preprocessed_v4v/'
    valid_frames = np.load(path + "valid_frames_face.npy")
    valid_BP = np.load(path + "valid_BP_mean.npy")
    valid_data = ((valid_frames[:, :, :, :3], valid_frames[:, :, :, -3:]), valid_BP)
    frames = np.load(path + data_type + '_frames_face.npy')
    BP_lf = np.load(path + data_type + '_BP_mean.npy')

    # Model setup
    img_rows = 48
    img_cols = 48
    frame_depth = 1
    input_shape = (img_rows, img_cols, 3)
    print('Using MTTS_CAN!')

    # Create a callback that saves the model's weights
    model = MTTS_CAN(frame_depth, nb_filters1, nb_filters2, input_shape,
                     dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2,
                     nb_dense=nb_dense)
    losses = tf.keras.losses.MeanAbsolutePercentageError()
    loss_weights = {"output_1": 1.0}
    opt = "adadelta"
    model.compile(loss=losses, loss_weights=loss_weights, optimizer=opt)
    if device_type == "local":
        path = "C:/Users/Zed/Desktop/Project-BMFG/BMFG/checkpoints/"
    else:
        path = "checkpoints/"
    if data_type == "test":
        model.load_weights(path + 'mtts_face.hdf5')
        model.evaluate(x=(frames[:, :, :, :3], frames[:, :, :, -3:]), y=BP_lf, batch_size=nb_batch)
    else:
        save_best_callback = ModelCheckpoint(filepath=path + 'mtts_face.hdf5',
                                             save_best_only=True, verbose=1)
        history = model.fit(x=(frames[:, :, :, :3], frames[:, :, :, -3:]), y=BP_lf, batch_size=nb_batch,
                            epochs=nb_epoch, callbacks=[save_best_callback],
                            verbose=1, shuffle=False, validation_data=valid_data,
                            use_multiprocessing=multiprocess)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', '--exp_type', type=str, default='video',
                        help='experiment type: model or video')
    parser.add_argument('-data', '--data_type', type=str, default='train',
                        help='data type')
    parser.add_argument('-device', '--device_type', type=str, default='local',
                        help='device type')
    parser.add_argument('-a', '--nb_filters1', type=int, default=32,
                        help='number of convolutional filters to use')
    parser.add_argument('-b', '--nb_filters2', type=int, default=64,
                        help='number of convolutional filters to use')
    parser.add_argument('-c', '--dropout_rate1', type=float, default=0.25,
                        help='dropout rates')
    parser.add_argument('-d', '--dropout_rate2', type=float, default=0.5,
                        help='dropout rates')
    parser.add_argument('-l', '--lr', type=float, default=1.0,
                        help='learning rate')
    parser.add_argument('-e', '--nb_dense', type=int, default=128,
                        help='number of dense units')
    parser.add_argument('-g', '--nb_epoch', type=int, default=24,
                        help='nb_epoch')
    parser.add_argument('--nb_batch', type=int, default=32,
                        help='nb_batch')
    parser.add_argument('--multiprocess', type=bool, default=False,
                        help='Use multiprocess or not')
    args = parser.parse_args()
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

    # if args.exp_type == "model":
    #     model_train(data_type=args.data_type, device_type=args.device_type,
    #                 task_num=0, nb_filters1=args.nb_filters1, nb_filters2=args.nb_filters2,
    #                 dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2,
    #                 nb_dense=args.nb_dense, nb_batch=args.nb_batch,
    #                 nb_epoch=args.nb_epoch, multiprocess=args.multiprocess)
    # else:
    #     data_processing_1(data_type=args.data_type, device_type=args.device_type)
    data_processing_3(data_type=args.data_type, device_type=args.device_type)