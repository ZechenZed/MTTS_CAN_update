import random
import numpy as np
import cv2
from skimage.util import img_as_float
from scipy.sparse import spdiags
import matplotlib.pyplot as plt

# def resize_image(img, h_new, w_old, h_old):
#     "I believe reszing image before face detection will speed up"
#     r = h_new / float(h_old)
#     dim = (int(w_old * r), h_new)
#     resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#     return resized

# def non_skin_remove(patches):
#     patches_hsv = [cv2.cvtColor(i, cv2.COLOR_RGB2HSV) for i in patches]
#     lower = np.array([0, 48, 80], dtype="uint8")
#     upper = np.array([20, 255, 255], dtype="uint8")
#     skinMask = [cv2.inRange(i, lower, upper) for i in patches_hsv]
#     skinindex = [i for i, j in enumerate(skinMask) if j.sum() > 5]
#     patches = [patches[i] for i in skinindex]
#     return patches


def preprocess_raw_video(videoFilePath, dim=48):
    # set up
    print("***********Processing "+videoFilePath[-12:]+"***********")
    t = []
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath)
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype=np.float32)
    success, img = vidObj.read()
    rows, cols, _ = img.shape
    height = rows
    width = cols
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Crop each frame size into dim x dim
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))

        # # Original From MTTS_CAN
        # vidLxL = cv2.resize(img_as_float(img[:, int(width / 2) - int(height / 2 + 1):int(height / 2) +
        #                                       int(width / 2), :]), (dim, dim), interpolation=cv2.INTER_AREA)

        # # Experimental
        # vidLxL = cv2.resize(img_as_float(img[200:1240, :, :]), (dim, dim), interpolation=cv2.INTER_AREA)

        # # Without considering the ratio
        #
        # vidLxL = cv2.resize(img_as_float(img[:, :, :]), (dim, dim), interpolation=cv2.INTER_AREA)

        # Face cropping
        # Add black edge around each frame of picture
        width_edge = 300
        height_edge = height * (width_edge / width)
        original_cf = np.float32([[0, 0], [width - 1, 0], [(width - 1) / 2, height - 1]])
        transed_cf = np.float32([[width_edge - 1, height_edge - 1], [width - width_edge - 1, height_edge - 1],
                                 [(width - 1) / 2, height - height_edge - 1]])
        matrix = cv2.getAffineTransform(original_cf, transed_cf)
        img = cv2.warpAffine(img, matrix, (cols, rows))

        # Face detection in gray scale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Cropping out ROI from the original image based on the "1:1:1"ish face ratio
        roi = 0
        for (x, y, w, h) in faces:
            roi = img_as_float(img[int(y - 0.25 * h):int(y + 1.05 * h), int(x - 0.15 * w):int(x + 1.15 * w), :])

        # Original resizing from MTTS_CAN
        vidLxL = cv2.resize(roi, (dim, dim), interpolation=cv2.INTER_AREA)

        # vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_CLOCKWISE)
        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1 / 255)] = 1 / 255
        Xsub[i, :, :, :] = vidLxL

        success, img = vidObj.read()
        i = i + 1
    # plt.imshow(Xsub[random.randint(0,i)])
    # plt.title('Sample Preprocessed Frame')
    # plt.show()
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


def count_frames(videoFilePath):
    print("***********Processing " + videoFilePath[-12:] + "***********")
    t = []
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath)
    success, img = vidObj.read()
    rows, cols, _ = img.shape
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))
        success, img = vidObj.read()
        i = i + 1
    normalized_len = len(t) - 1
    return normalized_len

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
