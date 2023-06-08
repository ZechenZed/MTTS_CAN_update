import os
import cv2
import glob
import json
import argparse
import numpy as np
from statistics import mean
from joblib import Parallel, delayed
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg', force=True)
import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D, Conv3D, Input, AveragePooling2D, \
    multiply, Dense, Dropout, Flatten, AveragePooling3D
from tensorflow.python.keras.models import Model

from skimage.util import img_as_float


class Attention_mask(tf.keras.layers.Layer):
    def call(self, x):
        xsum = K.sum(x, axis=1, keepdims=True)
        xsum = K.sum(xsum, axis=2, keepdims=True)
        xshape = K.int_shape(x)
        return x / xsum * xshape[1] * xshape[2] * 0.5

    def get_config(self):
        config = super(Attention_mask, self).get_config()
        return config


class TSM(tf.keras.layers.Layer):
    def call(self, x, n_frame, fold_div=3):
        nt, h, w, c = x.shape
        x = K.reshape(x, (-1, n_frame, h, w, c))
        fold = c // fold_div
        last_fold = c - (fold_div - 1) * fold
        out1, out2, out3 = tf.split(x, [fold, fold, last_fold], axis=-1)

        # Shift left
        padding_1 = tf.zeros_like(out1)
        padding_1 = padding_1[:, -1, :, :, :]
        padding_1 = tf.expand_dims(padding_1, 1)
        _, out1 = tf.split(out1, [1, n_frame - 1], axis=1)
        out1 = tf.concat([out1, padding_1], axis=1)

        # Shift right
        padding_2 = tf.zeros_like(out2)
        padding_2 = padding_2[:, 0, :, :, :]
        padding_2 = tf.expand_dims(padding_2, 1)
        out2, _ = tf.split(out2, [n_frame - 1, 1], axis=1)
        out2 = tf.concat([padding_2, out2], axis=1)

        out = tf.concat([out1, out2, out3], axis=-1)
        out = K.reshape(out, (-1, h, w, c))

        return out

    def get_config(self):
        config = super(TSM, self).get_config()
        return config


def TSM_Cov2D(x, n_frame, nb_filters=128, kernel_size=(3, 3), activation='tanh', padding='same'):
    x = TSM()(x, n_frame)
    x = Conv2D(nb_filters, kernel_size, padding=padding, activation=activation)(x)
    return x


def MTTS_CAN(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25,
             dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128):
    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = TSM_Cov2D(diff_input, n_frame, nb_filters1, kernel_size, padding='same', activation='tanh')
    d2 = TSM_Cov2D(d1, n_frame, nb_filters1, kernel_size, padding='valid', activation='tanh')

    r1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    d3 = AveragePooling2D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)

    r3 = AveragePooling2D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)

    d5 = TSM_Cov2D(d4, n_frame, nb_filters2, kernel_size, padding='same', activation='tanh')
    d6 = TSM_Cov2D(d5, n_frame, nb_filters2, kernel_size, padding='valid', activation='tanh')

    r5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])

    d7 = AveragePooling2D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)

    d10_y = Dense(nb_dense, activation='tanh')(d9)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(1, name='output_1')(d11_y)

    #     d10_r = Dense(nb_dense, activation='tanh')(d9)
    #     d11_r = Dropout(dropout_rate2)(d10_r)
    #     out_r = Dense(1, name='output_2')(d11_r)

    #     model = Model(inputs=[diff_input, rawf_input], outputs=[out_y, out_r])
    model = Model(inputs=[diff_input, rawf_input], outputs=[out_y])
    return model


def preprocess_raw_video(videoFilePath, dim=36):
    # set up
    t = []
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath)
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype=np.float32)
    success, img = vidObj.read()
    rows, cols, _ = img.shape
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Crop each frame size into dim x dim
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))
        vidLxL = cv2.resize(img_as_float(img[:, :, :]), (dim, dim), interpolation=cv2.INTER_AREA)
        vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_CLOCKWISE)
        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1 / 255)] = 1 / 255
        Xsub[i, :, :, :] = vidLxL

        success, img = vidObj.read()
        i = i + 1

    # Normalized Frames in the motion branch
    normalized_len = len(t) - 1
    dXsub = np.zeros((normalized_len, dim, dim, 3), dtype=np.float32)
    for j in range(normalized_len - 1):
        dXsub[j, :, :, :] = (Xsub[j + 1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j + 1, :, :, :] + Xsub[j, :, :, :])
    dXsub = dXsub / np.std(dXsub)

    # Normalize raw frames in the apperance branch
    Xsub = Xsub - np.mean(Xsub)
    Xsub = Xsub / np.std(Xsub)
    Xsub = Xsub[:totalFrames - 1, :, :, :]
    # Plot an example of data after preprocess
    dXsub = np.concatenate((dXsub, Xsub), axis=3)
    return dXsub


if __name__ == "__main__":
    # Path setting
    # video_train_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Videos/train/"
    # video_valid_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Videos/valid/"
    # video_test_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase2_data/Videos/test/"
    # BP_phase1_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase1_data/Ground_truth/BP_raw_1KHz/"
    # BP_test_path = "C:/Users/Zed/Desktop/Project-BMFG/Phase2_data/blood_pressure/test_set_bp/"

    video_train_path = "../../../../edrive2/zechenzh/V4V/Phase1_data/Videos/train/"
    video_valid_path = "../../../../edrive2/zechenzh/V4V/Phase1_data/Videos/valid/"
    video_test_path = "../../../../edrive2/zechenzh/V4V/Phase2_data/Videos/test/"
    BP_phase1_path = "../../../../edrive2/zechenzh/V4V/Phase1_data/Ground_truth/BP_raw_1KHz/"
    BP_test_path = "../../../../edrive2/zechenzh/V4V/Phase2_data/blood_pressure/test_set_bp/"

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', '--exp_type', type=str, default='test',
                        help='experiment type')
    parser.add_argument('-t', '--task', type=int, default=0,
                        help='the order of exp')
    parser.add_argument('-i', '--data_dir', type=str,
                        help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='./rPPG-checkpoints',
                        help='Location for parameter checkpoints and samples')
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
    args = parser.parse_args()
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

    # # Check GPU
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    # tf.keras.backend.clear_session()
    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    ################################## Train Dataset ###########################################
    # Video path reading
    train_videos = []
    for path in os.listdir(video_train_path):
        if os.path.isfile(os.path.join(video_train_path, path)):
            train_videos.append(path)
    num_video = len(train_videos)
    print(num_video,train_videos)

    # Video Preprocessing
    videos = [Parallel(n_jobs=4)(
        delayed(preprocess_raw_video)(video_train_path + video) for video in train_videos)]
    videos = videos[0]
    tt_frame = 0
    for i in range(num_video):
        tt_frame += videos[i].shape[0] // 10 * 10

    # BP reading and processing
    BP_train_path = []
    for path in os.listdir(BP_phase1_path):
        if os.path.isfile(os.path.join(BP_phase1_path, path)):
            BP_train_path.append(path)

    frames = np.zeros(shape=(tt_frame, 36, 36, 6))
    BP_lf = np.zeros(shape=tt_frame)
    frame_ind = 0
    for j in range(num_video):
        temp = np.loadtxt(BP_phase1_path + BP_train_path[j])
        cur_frames = videos[j].shape[0] // 10 * 10
        temp_lf = np.zeros(cur_frames)
        frames[frame_ind:frame_ind + cur_frames, :, :, :] = videos[j][0:cur_frames, :, :, :]
        for i in range(0, cur_frames):
            temp_lf[i] = mean(temp[i * 40:(i + 1) * 40])
        BP_lf[frame_ind:frame_ind + cur_frames] = temp_lf
        frame_ind += cur_frames
    np.save('../../../../edrive2/zechenzh/preprocessed_v4v/train_frames.npy', frames)
    np.save('../../../../edrive2/zechenzh/preprocessed_v4v/train_BP.npy', BP_lf)

    ########################################## Test Dataset ################################################
    # # Video path reading
    # test_videos = []
    # for path in os.listdir(video_test_path):
    #     if os.path.isfile(os.path.join(video_test_path, path)):
    #         test_videos.append(path)
    # num_video = len(test_videos)
    # # print(num_video,train_videos)
    #
    # # Video Preprocessing
    # videos = [Parallel(n_jobs=-1)(
    #     delayed(preprocess_raw_video)(video_test_path + video) for video in test_videos)]
    # videos = videos[0]
    # tt_frame = 0
    # for i in range(num_video):
    #     tt_frame += videos[i].shape[0] // 10 * 10
    #
    # # BP reading and processing
    # BP_test = []
    # for path in os.listdir(BP_test_path):
    #     if os.path.isfile(os.path.join(BP_test_path, path)):
    #         BP_test.append(path)
    #
    # frames = np.zeros(shape=(tt_frame, 36, 36, 6))
    # BP_lf = np.zeros(shape=tt_frame)
    # frame_ind = 0
    # for j in range(num_video):
    #     temp = np.loadtxt(BP_test_path + BP_test[j])
    #     cur_frames = videos[j].shape[0] // 10 * 10
    #     temp_lf = np.zeros(cur_frames)
    #     frames[frame_ind:frame_ind + cur_frames, :, :, :] = videos[j][0:cur_frames, :, :, :]
    #     for i in range(0, cur_frames):
    #         temp_lf[i] = mean(temp[i * 40:(i + 1) * 40])
    #     BP_lf[frame_ind:frame_ind + cur_frames] = temp_lf
    #     frame_ind += cur_frames

    # np.save('test_frames.npy', frames)
    # np.save('test_BP.npy', BP_lf)
    # frames = np.load('../../preprocessed_v4v/train_frames.npy')
    # BP_lf = np.load('../../preprocessed_v4v/train_BP.npy')
    # frames = np.load('../../preprocessed_v4v/test_frames.npy')
    # BP_lf = np.load('../../preprocessed_v4v/test_BP.npy')
    # # Train 132505 * 6
    # frames = frames[132505*5:132505*6]
    # BP_lf = BP_lf[132505*5:132505*6]
    # Test 102090 * 4
    # frames = frames[102090*3:102090*4]
    # BP_lf = BP_lf[102090*3:102090*4]
    # BP_lf = BP_lf[0:1610]
    # plt.plot(BP_lf, label="BP downsampled into lower freq")
    # plt.legend()
    # plt.show()
    # for i in range(6):
    #     np.save('../../preprocessed_v4v/train_frames_' + str(i) + '.npy', frames[132505 * i:132505 * (i + 1)])
    #     np.save('../../preprocessed_v4v/train_BP_' + str(i) + '.npy', BP_lf[132505 * i:132505 * (i + 1)])

    # # Model setup
    # img_rows = 36
    # img_cols = 36
    # frame_depth = 1
    # input_shape = (img_rows, img_cols, 3)
    # print('Using MTTS_CAN!')
    #
    # # Create a callback that saves the model's weights
    # model = MTTS_CAN(frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
    #                  dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
    # losses = tf.keras.losses.MeanAbsoluteError()
    # loss_weights = {"output_1": 1.0}
    # opt = "adadelta"
    # model.compile(loss=losses, loss_weights=loss_weights, optimizer=opt)
    # if args.exp_type == "test":
    #     model.load_weights('../checkpoints/my_mtts.hdf5')
    #     model.evaluate(x=(frames[:, :, :, :3], frames[:, :, :, -3:]), y=BP_lf, batch_size=32)
    # else:
    #     save_best_callback = ModelCheckpoint(filepath="../checkpoints/my_mtts.hdf5", save_best_only=True, verbose=1)
    #     early_stop = tf.keras.callbacks.EarlyStopping(monitor=losses, patience=10)
    #     history = model.fit(x=(frames[:, :, :, :3], frames[:, :, :, -3:]), y=BP_lf, validation_split=0.2,
    #                         batch_size=128,
    #                         epochs=20, callbacks=[save_best_callback, early_stop], verbose=1, shuffle=False)
