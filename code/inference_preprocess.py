import random

import numpy as np
import cv2
from skimage.util import img_as_float
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import scipy.io
from scipy.sparse import spdiags


def resize_image(img, h_new, w_old, h_old):
    "I believe reszing image before face detection will speed up"
    r = h_new / float(h_old)
    dim = (int(w_old * r), h_new)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def non_skin_remove(patches):
    patches_hsv = [cv2.cvtColor(i, cv2.COLOR_RGB2HSV) for i in patches]
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinMask = [cv2.inRange(i, lower, upper) for i in patches_hsv]
    skinindex = [i for i, j in enumerate(skinMask) if j.sum() > 5]
    patches = [patches[i] for i in skinindex]
    return patches


def preprocess_raw_video(videoFilePath, dim=36):
    #########################################################################
    # set up
    t = []
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath)
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))  # get total frame size
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype=np.float32)
    height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
    success, img = vidObj.read()
    rows, cols, _ = img.shape
    # print("image shape",img.shape)
    # print("Orignal Height", height)
    # print("Original width", width)
    # print("Total number of frames:", totalFrames)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    #########################################################################
    # Crop each frame size into dim x dim
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))  # current timestamp in milisecond
        # vidLxL = cv2.resize(
        #     img_as_float(img[:, int(width / 2) - int(height / 2 + 1):int(height / 2) + int(width / 2), :]), (dim, dim),
        #     interpolation=cv2.INTER_AREA)

        # TODO: Find a new way to crop the facial area for V4V dataset

        ## Experimental
        # vidLxL = cv2.resize(img_as_float(img[200:1240, :, :]), (dim, dim), interpolation=cv2.INTER_AREA)

        # Face Crop
        # img = resize_image(img, 300, width, height)
        # width, height, _ = img.shape

        width_edge = 200
        height_edge = height * (width_edge / width)
        original_cf = np.float32([[0, 0], [width - 1, 0], [(width - 1) / 2, height - 1]])
        transed_cf = np.float32([[width_edge - 1, height_edge - 1], [width - width_edge - 1, height_edge - 1],
                                 [(width - 1) / 2, height - height_edge - 1]])
        matrix = cv2.getAffineTransform(original_cf, transed_cf)
        img = cv2.warpAffine(img, matrix, (cols, rows))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        roi = 0
        # print(img.shape)

        # if faces == ():
        #     print("WARNING")
        #     roi = img_as_float(img[int(200 + height_edge):int(1240 - height_edge), int(100):int(width - 100), :])
        # else:
        #     for (x, y, w, h) in faces:
        #         roi = img_as_float(img[y - 200:y + w, x - 100:x + w + 100, :])
        for (x, y, w, h) in faces:
            roi = img_as_float(img[y - 170:y + w -10, x - 80:x + w + 80, :])

        vidLxL = cv2.resize(roi, (dim, dim), interpolation=cv2.INTER_AREA)
        vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_CLOCKWISE)  # rotate 90 degree

        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        # vidLxL = cv2.cvtColor(vidLxL, cv2.COLOR_BGR2RGB)

        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1 / 255)] = 1 / 255
        Xsub[i, :, :, :] = vidLxL
        # Xsub[i, :, :, :] = vidLxL / 255.0

        success, img = vidObj.read()  # read the next one
        i = i + 1


    #########################################################################
    # Normalized Frames in the motion branch
    normalized_len = len(t) - 1
    dXsub = np.zeros((normalized_len, dim, dim, 3), dtype=np.float32)
    for j in range(normalized_len - 1):
        dXsub[j, :, :, :] = (Xsub[j + 1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j + 1, :, :, :] + Xsub[j, :, :, :])
    dXsub = dXsub / np.std(dXsub)
    #########################################################################
    # Normalize raw frames in the apperance branch
    Xsub = Xsub - np.mean(Xsub)
    Xsub = Xsub / np.std(Xsub)
    Xsub = Xsub[:totalFrames - 1, :, :, :]
    #########################################################################
    # Plot an example of data after preprocess
    dXsub = np.concatenate((dXsub, Xsub), axis=3)
    return dXsub


def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal
