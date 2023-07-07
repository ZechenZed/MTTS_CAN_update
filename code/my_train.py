import os
import json
import argparse
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from inference_preprocess import preprocess_raw_video, count_frames
from model import MTTS_CAN, MT_CAN_3D

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"


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
    video_file_path = video_file_path[0:10]
    num_video = len(video_file_path)
    print(num_video)

    videos = [Parallel(n_jobs=12)(
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
        np.save('/edrive2/zechenzh/preprocessed_v4v/' + data_type + '_frames_ratio.npy', frames)
        np.save('/edrive2/zechenzh/preprocessed_v4v/' + data_type + '_BP_mean.npy', BP_lf)
    else:
        np.save('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/' + data_type + '_frames_ratio.npy', frames)
        np.save('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/' + data_type + '_BP_mean.npy', BP_lf)


# Video --> 1000Hz(Not recommended)
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
    # video_file_path = video_file_path[0:2]
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


def new_data_process(data_type, device_type, dim=48, image=str()):
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
    video_file_path = video_file_path[0:5]
    num_video = len(video_file_path)
    print('Processing ' + str(num_video) + ' Videos')

    videos = [Parallel(n_jobs=12)(
        delayed(preprocess_raw_video)(video_folder_path + video) for video in video_file_path)]
    videos = videos[0]

    # Max Frame finding
    max_frame = 0
    for i in range(num_video):
        max_frame = max(max_frame, videos[i].shape[0] // 10 * 10)
    videos_batch = np.zeros((num_video, max_frame, 48, 48, 6))

    # BP file finding
    BP_file_path = []
    for path in sorted(os.listdir(BP_folder_path)):
        if os.path.isfile(os.path.join(BP_folder_path, path)):
            BP_file_path.append(path)

    BP_lf = np.zeros((num_video, max_frame))
    for i in range(num_video):
        temp_BP = np.loadtxt(BP_folder_path + BP_file_path[i])
        current_frames = videos[i].shape[0] // 10 * 10
        temp_BP_lf = np.zeros(current_frames)
        for j in range(0, current_frames):
            temp_BP_lf[j] = mean(temp_BP[j * 40:(j + 1) * 40])

        # Systolic BP finding and linear interp
        temp_BP_lf_systolic, _ = find_peaks(temp_BP_lf, distance=10)
        temp_BP_lf_systolic_inter = np.zeros(current_frames)
        prev_index = 0
        for index in temp_BP_lf_systolic:
            y_interp = interp1d([prev_index, index], [temp_BP_lf[prev_index], temp_BP_lf[index]])
            for k in range(prev_index, index + 1):
                temp_BP_lf_systolic_inter[k] = y_interp(k)
            prev_index = index
        y_interp = interp1d([prev_index, current_frames - 1], [temp_BP_lf[prev_index], temp_BP_lf[current_frames - 1]])
        for l in range(prev_index, current_frames):
            temp_BP_lf_systolic_inter[l] = y_interp(l)

        BP_lf[i][0:current_frames] = temp_BP_lf_systolic_inter[:]
        plt.plot(temp_BP_lf_systolic_inter)
        plt.show()
        videos_batch[i][0:current_frames, :, :, :] = videos[i][0:current_frames, :, :, :]

    saving_path = str()
    if device_type == "remote":
        saving_path = '/edrive2/zechenzh/preprocessed_v4v_3d/'
    else:
        saving_path = 'C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v_3d/'
    np.save(saving_path + data_type + '_frames_3d_'+image+'.npy', videos_batch)
    np.save(saving_path + data_type + '_BP_3d_systolic.npy', BP_lf)



def BP_systolic(data_type, device_type):
    if device_type == "local":
        BP_folder_path = "C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/"
    else:
        BP_folder_path = "/edrive2/zechenzh/preprocessed_v4v/"

    BP = np.load(BP_folder_path + data_type + '_BP_mean.npy')
    size = BP.shape[0]
    BP_systolic, _ = find_peaks(BP, distance=10)
    BP_inter = np.zeros(size)
    prev_index = 0
    for index in BP_systolic:
        y_interp = interp1d([prev_index, index], [BP[prev_index], BP[index]])
        for i in range(prev_index, index + 1):
            BP_inter[i] = y_interp(i)
        prev_index = index
    y_interp = interp1d([prev_index, size - 1], [BP[prev_index], BP[size - 1]])
    for i in range(prev_index, size):
        BP_inter[i] = y_interp(i)

    # plt.plot(BP, label='Original')
    # plt.plot(BP_inter, label='peaks')
    # plt.show()

    # Saving processed frames
    if device_type == "remote":
        np.save('/edrive2/zechenzh/preprocessed_v4v/' + data_type + '_BP_mean_systolic.npy', BP_inter)
    else:
        np.save('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/' + data_type + '_BP_mean_systolic.npy', BP_inter)


def model_train(data_type, device_type, task_num, nb_filters1, nb_filters2,
                dropout_rate1, dropout_rate2, nb_dense, nb_batch, nb_epoch, multiprocess):
    path = ""
    if device_type == "local":
        path = 'C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/'
    else:
        path = '/edrive2/zechenzh/preprocessed_v4v/'
    valid_frames = np.load(path + "valid_frames_face.npy")
    valid_BP = np.load(path + "valid_BP_mean_systolic.npy")
    valid_data = ((valid_frames[:, :, :, :3], valid_frames[:, :, :, -3:]), valid_BP)
    frames = np.load(path + data_type + '_frames_face.npy')
    BP_lf = np.load(path + data_type + '_BP_mean_systolic.npy')

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
    losses = tf.keras.losses.MeanAbsoluteError()
    loss_weights = {"output_1": 1.0}
    opt = "Adam"
    model.compile(loss=losses, loss_weights=loss_weights, optimizer=opt)
    if device_type == "local":
        path = "C:/Users/Zed/Desktop/Project-BMFG/BMFG/checkpoints/"
    else:
        path = "/home/zechenzh/checkpoints/"
    if data_type == "test":
        model.load_weights(path + 'mtts_sys_kernal99_face_drop2_nb256.hdf5')
        model.evaluate(x=(frames[:, :, :, :3], frames[:, :, :, -3:]), y=BP_lf, batch_size=nb_batch)
    else:
        # early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        save_best_callback = ModelCheckpoint(filepath=path + 'mtts_sys_kernal99_face_drop2_nb256.hdf5',
                                             save_best_only=True, verbose=1)
        model.fit(x=(frames[:, :, :, :3], frames[:, :, :, -3:]), y=BP_lf, batch_size=nb_batch,
                  epochs=nb_epoch, callbacks=[save_best_callback],
                  verbose=1, shuffle=False, validation_data=valid_data,
                  use_multiprocessing=multiprocess)


def new_model_train(data_type, device_type, nb_filters1, nb_filters2,
                    dropout_rate1, dropout_rate2, nb_dense, nb_batch, nb_epoch, multiprocess):
    # path = ""
    # if device_type == "local":
    #     path = 'C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/'
    # else:
    #     path = '/edrive2/zechenzh/preprocessed_v4v_3d/'

    # frames = np.load('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/train_frames_previous.npy')
    # BP_lf = np.load('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/train_BP_mean.npy')
    # BP_lf = BP_lf[0:1000]

    # frames = np.load('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/train_frames_test.npy')
    # BP_lf = np.load('C:/Users/Zed/Desktop/Project-BMFG/preprocessed_v4v/train_BP_test.npy')

    frames = np.load('/edrive2/zechenzh/preprocessed_v4v_3d/train_frames_3d_face_large.npy')
    BP_lf = np.load('/edrive2/zechenzh/preprocessed_v4v_3d/train_3d_systolic.npy')

    # Model setup
    img_rows = 48
    img_cols = 48
    frame_depth = 1610
    input_shape = (frame_depth,img_rows, img_cols, 3)
    print('Using MT_CAN_3D!')

    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        # Create a callback that saves the model's weights
        model = MT_CAN_3D(frame_depth, nb_filters1, nb_filters2, input_shape,
                         dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2,
                         nb_dense=nb_dense)
        losses = tf.keras.losses.MeanAbsoluteError()
        loss_weights = {"output_1": 1.0}
        opt = "Adam"
        model.compile(loss=losses, loss_weights=loss_weights, optimizer=opt)

    if device_type == "local":
        path = "C:/Users/Zed/Desktop/Project-BMFG/BMFG/checkpoints/"
    else:
        path = "/home/zechenzh/checkpoints/"
    if data_type == "test":
        model.load_weights(path + 'mt3d.hdf5')
        model.evaluate(x=(frames[:, :, :, :, :3], frames[:, :, :, :, -3:]), y=BP_lf, batch_size=[nb_batch,5])
    else:
        # early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        save_best_callback = ModelCheckpoint(filepath=path + 'mt3d.hdf5',
                                             save_best_only=True, verbose=1)
        model.fit(x=(frames[:, :, :, :, :3], frames[:, :, :, :, -3:]), y=BP_lf, batch_size=nb_batch,
                  epochs=nb_epoch, callbacks=[save_best_callback],
                  verbose=1, shuffle=False, use_multiprocessing=multiprocess)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', '--exp_type', type=str, default='video',
                        help='experiment type: model or video')
    parser.add_argument('-data', '--data_type', type=str, default='train',
                        help='data type')
    parser.add_argument('-image', '--image_type', type=str, default='face_large',
                        help='input image area')
    parser.add_argument('-device', '--device_type', type=str, default='local',
                        help='device type')
    parser.add_argument('-a', '--nb_filters1', type=int, default=32,
                        help='number of convolutional filters to use')
    parser.add_argument('-b', '--nb_filters2', type=int, default=64,
                        help='number of convolutional filters to use')
    parser.add_argument('-c', '--dropout_rate1', type=float, default=0.125,
                        help='dropout rates')
    parser.add_argument('-d', '--dropout_rate2', type=float, default=0.5,
                        help='dropout rates')
    parser.add_argument('-l', '--lr', type=float, default=1.0,
                        help='learning rate')
    parser.add_argument('-e', '--nb_dense', type=int, default=256,
                        help='number of dense units')
    parser.add_argument('-g', '--nb_epoch', type=int, default=12,
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


    new_data_process(data_type=args.data_type, device_type=args.device_type, image=args.image_type)

    # new_model_train(data_type=args.data_type, device_type=args.device_type,
    #                 nb_filters1=args.nb_filters1, nb_filters2=args.nb_filters2,
    #                 dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2,
    #                 nb_dense=args.nb_dense, nb_batch=args.nb_batch,
    #                 nb_epoch=args.nb_epoch, multiprocess=args.multiprocess)
