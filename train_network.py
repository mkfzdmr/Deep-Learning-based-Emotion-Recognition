from __future__ import print_function
import pandas as pd
import numpy as np
import scipy.stats as scs
import re
import natsort
from imutils import paths
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from numpy import savetxt
from numpy import genfromtxt
import csv

from keras.utils import np_utils

pd.options.display.max_columns = None
pd.options.display.precision = 4
# import the necessary packages
from keras.optimizers import Adam

from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from model_1 import networkArchFonc
from imutils import paths
import matplotlib.pyplot as plt

import argparse
import random
# from pandas_ml import ConfusionMatrix
import cv2
import os
import pandas as pd
import scikitplot as skplt
# import imutils
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix, balanced_accuracy_score, precision_recall_curve, matthews_corrcoef, roc_curve, jaccard_score, \
    hamming_loss, fbeta_score, precision_recall_fscore_support, zero_one_loss, average_precision_score
from sklearn.metrics import cohen_kappa_score, roc_auc_score, mean_squared_error, auc
from inspect import signature
from sklearn.model_selection import train_test_split
import time

np.random.seed(1234)
from functools import reduce
import math as m

import scipy.io
# import theano
# import theano.tensor as T

from scipy.interpolate import griddata
from sklearn.preprocessing import scale

sira = '52'

# gama = (30,45)
# alpha = (8,13)
# beta = (14,30)
resim_boyut = 16
dataset = 'relabeled_data'
etiket = 'labels/dominance.csv'
frame_duration = 15
overlap = 0  # degisecek
batch_size = 64
num_classes = 2
epochs = 400
model_save = 'modeller/m' + sira
test_sonuc = "sonuclar/sonuc" + sira
test_sonuc2 = "sonuclar/confision" + sira
PR = "sonuclar/PR-Grafik" + sira
roc = "sonuclar/roc" + sira


# sayac =0


def fft(snippet):
    Fs = 128.0  # sampling rate
    # Ts = len(snippet)/Fs/Fs; # sampling interval
    snippet_time = len(snippet) / Fs
    Ts = 1.0 / Fs  # sampling interval
    t = np.arange(0, snippet_time, Ts)  # time vector

    # ff = 5;   # frequency of the signal
    # y = np.sin(2*np.pi*ff*t)
    y = snippet
    #     print('Ts: ',Ts)
    #     print(t)
    #     print(y.shape)
    n = len(y)  # length of the signal
    k = np.arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    frq = frq[range(n // 2)]  # one side frequency range

    #   Y = np.fft.fft(y)/n # fft computing and normalization
    # ydeneme=np.fft.fft(y)

    Y = np.fft.fft(y)
    Y = abs(Y)
    # Y=np.square(Y)
    Y = Y / n

    Y = Y[range(n // 2)]
    # ydeneme =ydeneme[range(n // 2)]
    # plt.plot(frq, Y)
    # plt.show()
    # plt.plot(frq, ydeneme)
    # plt.show()
    # plt.plot(frq, abs(Y))
    # plt.show()

    # Added in: (To remove bias.)
    # Y[0] = 0
    # return frq,abs(Y)
    return frq, Y


def gama_alpha_beta_averages(f, Y):
    gama_range = (30, 45)
    alpha_range = (8, 13)
    beta_range = (14, 30)
    # gama1 = Y[(f > gama_range[0]) & (f <= gama_range[1])].sum()
    gama = Y[(f > gama_range[0]) & (f <= gama_range[1])].mean()
    alpha = Y[(f > alpha_range[0]) & (f <= alpha_range[1])].mean()
    beta = Y[(f > beta_range[0]) & (f <= beta_range[1])].mean()

    return gama, alpha, beta


def cart2sph(x, y, z):
    x2_y2 = x ** 2 + y ** 2
    r = m.sqrt(x2_y2 + z ** 2)  # r
    elev = m.atan2(z, m.sqrt(x2_y2))  # Elevation
    az = m.atan2(y, x)  # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    return rho * m.cos(theta), rho * m.sin(theta)


def steps_m(samples, frame_duration, overlap):
    Fs = 128
    i = 0
    intervals = []
    samples_per_frame = Fs * frame_duration
    while i + samples_per_frame <= samples:
        intervals.append((i, i + samples_per_frame))
        i = i + samples_per_frame - int(samples_per_frame * overlap)
    return intervals


def aep_frame_maker(df, frame_duration):
    Fs = 128.0
    frame_length = Fs * frame_duration
    frames = []
    steps = steps_m(len(df), frame_duration, overlap)
    for i, _ in enumerate(steps):
        frame = []
        # if i == 0:
        # continue
        # else:
        for channel in df.columns:
            snippet = np.array(df.loc[steps[i][0]:steps[i][1], int(channel)])
            f, Y = fft(snippet)  # real part fft bul
            gama, alpha, beta = gama_alpha_beta_averages(f, Y)
            # plt.plot(f, Y)
            # plt.show()
            frame.append([gama, alpha, beta])
            # plt.plot(frame[0])
            # plt.show()

        # global sayac
        # if sayac==10:
        #     for k in frame:
        #       a=  sum(k)/3
        #       powerlist.append(a)
        # elif sayac==92:
        #     for k in frame:
        #         a = sum(k) / 3
        #         powerlist2.append(a)

        # sayac = sayac + 1

        frames.append(frame)
        # plt.plot(frames[0])
        # plt.show()
    return np.array(frames)


# location read
results = []
with open("loc2d.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
    for row in reader:  # each row is a list
        results.append(np.array(row))
        # print(row)

locs_2d = np.array(results)


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]  # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] // nElectrodes
    for c in range(int(n_colors)):
        feat_array_temp.append(features[:, c * nElectrodes: nElectrodes * (c + 1)])

    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints * 1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints * 1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
    # Interpolating
    for i in range(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                               method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i + 1, nSamples), end='\r')
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)  # swap axes to have [samples, colors, W, H]


def data_marker(file_names, labels, image_size, frame_duration, overlap):
    Fs = 128.0  # sampling rate
    frame_length = Fs * frame_duration

    print('Generating training data...')

    for i, file in enumerate(file_names):
        print('Processing session: ', file, '. (', i + 1, ' of ', len(file_names), ')')
        data = genfromtxt(file, delimiter=',').T
        df = pd.DataFrame(data)

        X_0 = aep_frame_maker(df, frame_duration)
        # steps = np.arange(0,len(df),frame_length)
        X_1 = X_0.reshape(len(X_0), 32 * 3)

        images = gen_images(np.array(locs_2d), X_1, image_size, normalize=False)
        images = np.swapaxes(images, 1, 3)
        print(len(images), ' frames generated with label ', labels[i], '.')
        print('\n')
        if i == 0:
            X = images
            y = np.ones(len(images)) * labels[0]
        else:
            X = np.concatenate((X, images), axis=0)
            y = np.concatenate((y, np.ones(len(images)) * labels[i]), axis=0)

    return X, np.array(y)


# def build(inputShape, classes):
#     # initialize the model
#     model = Sequential()
#     # inputShape = (height, width, depth)
#
#     # if we are using "channels first", update the input shape
#     # if K.image_data_format() == "channels_first":
#     #     inputShape = (depth, height, width)
#
#     model = Sequential([
#         Conv2D(filters=32,
#                                kernel_size=5,
#                                strides=2,
#                                padding="same",
#                                activation="relu",
#                                input_shape=inputShape),
#          MaxPooling2D(pool_size=2, strides=2, padding="same"),
#          Conv2D(filters=64, kernel_size=5, strides=2, padding="same", activation="relu"),
#          MaxPooling2D(pool_size=2, strides=2, padding="same"),
#          Conv2D(filters=128, kernel_size=5, strides=2, padding="same", activation="relu"),
#          MaxPooling2D(pool_size=2, strides=2, padding="same"),
#          Reshape((128, 3), name='predictions'),
#          LSTM(128, return_sequences=True),
#          LSTM(128, return_sequences=True),
#          Flatten(),
#          Dense(100, activation="relu"),
#
#          Dense(1, activation='sigmoid')
#     ])
#     return model

imagePaths = sorted(list(paths.list_files(dataset)))
imagePaths = (natsort.natsorted(imagePaths))
# print(imagePaths)
file_names = imagePaths

# label read as array
# with open('labels/arousal.csv') as f:
with open(etiket) as f:
    output = [float(s) for line in f.readlines() for s in line[:-1].split(',')]
    output = [round(x) for x in output]
    # print(output)

labels = output
image_size = resim_boyut

X, y = data_marker(file_names, labels, image_size, frame_duration, overlap)

# #cizim isleri
print(X.shape)
print(y.shape)

# resim10=X[10]
# resim47=X[47]
# print(sayac)
# print(powerlist)
# print("digeri \n")
# print(powerlist2)

# savetxt('powerlist.csv', powerlist, delimiter=',')
# savetxt('powerlist2.csv', powerlist2, delimiter=',')
# plt.figure()
# plt.imshow(resim10) # 5.csv arousal 1
# plt.show()
#
# plt.savefig("res10")
# plt.figure()
# plt.imshow(resim47) # 7.csv arousal 1
# plt.show()
# plt.savefig("res47")
#
# plt.imshow(X[9]) #  27.csv arousal 0
# plt.show()
# plt.imshow(X[8]) # 42.csv arousal 1
# plt.show()
# plt.imshow(X[17]) # 43.csv arousal 1
# plt.show()
# plt.imshow(X[35]) # 47.csv arousal 1
# plt.show()


# print("x boyut: ", X.shape())


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# input image dimensions
img_rows, img_cols = resim_boyut, resim_boyut

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

input_shape = (img_rows, img_cols, 3)

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# model = Sequential()
# model.add(Conv2D(16, (5, 5), padding='same', input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# # model.add(Dropout(0.25))
#
#
# model.add(Conv2D(32, (5, 5), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# model.add(Conv2D(64, (2, 2), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#

#
#
#
# model.add(Conv2D(64, (2, 2), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
# model.add(Conv2D(128, (2, 2), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
# model.add(Conv2D(32, (2, 2), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (2, 2), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
# model.add(Conv2D(64, (2, 2), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (2, 2), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
# model.add(Conv2D(128, (2, 2), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(128, (2, 2), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#


# model = Sequential()
# model.add(Conv2D(32, (2, 2), padding='same',input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(Dropout(0.5)) #kapalı
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(64, (2, 2), padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.5)) #kapalı
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(128, (2, 2), padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.5)) #kapalı
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


# model.add(Conv2D(32, (4, 4)))
# model.add(Activation('relu'))
# #model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))
# # model.add(Conv2D(32, (4, 4)))
# model.add(Activation('relu'))
# #model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# #model.add(Conv2D(128, (2, 2)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (2, 2)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))


# model.add(Flatten())
# model.add(Dense(100))  # 100dü
# model.add(Activation('relu'))
# # model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))


model = networkArchFonc.build(width=resim_boyut, height=resim_boyut, depth=3, classes=2)
opt = Adam(lr=0.001, decay=1e-6)

# opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss="binary_crossentropy",
              metrics=["accuracy"])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

H = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True, verbose=2)

# save the model to disk
print("[INFO] saving model file...")
model.save(model_save)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="training_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="training_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Performance Metrics of Valence Emotion State")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="best", bbox_to_anchor=(0.5, 0., 0.5, 0.5))
plt.savefig(test_sonuc, dpi=500)

##################################################################
# plt.cla()
# plt.clf()
#
# plt.style.use("ggplot")
# plt.figure()
# N = EPOCHS
# #plt.plot(np.arange(0, N), H.history["loss"], label="training_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# #plt.plot(np.arange(0, N), H.history["accuracy"], label="training_acc")
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
# plt.title("Validation Performance Metrics of Arousal")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="best", bbox_to_anchor=(0.5, 0., 0.5, 0.5))
# plt.savefig(test_sonuc3, dpi=500)


plt.cla()
plt.clf()

# Plot Confusion Matrix
# ben bunu trainle  denicem. TesX testY yerine trainX trainY yaz
Y_pred = model.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
etiket = ["UNLIKE", "LIKE", ]  # LOW HIGH
# etiket = ["UNLIKE", "LIKE", ] #LOW HIGH

confusion_mtx = confusion_matrix(y_true, y_pred)
# plot the confusion matrix
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, fmt=".1f", linewidths=0.01, cmap="Blues", linecolor="gray", ax=ax)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

ax.set_xticklabels(etiket)
ax.set_yticklabels(etiket)
# ax.set_ylim(len(confusion_mtx)+0.5, -0.5) # taşma probleminden doalyı ekledim
plt.savefig(test_sonuc2, dpi=500)
print(H.history.keys())
print(confusion_mtx)


# plot second confusion matrix
# plot_model(model,to_file=model_File,show_shapes=True,show_layer_names=True)
def cm_analysis(y_true, y_pred, filename, etikets, ymap=None, figsize=(10, 10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in etikets]
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=etikets, columns=etikets)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    plt.title("Confusion Matrix")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap=sns.cm.rocket_r)
    plt.title("Confusion Matrix")
    plt.savefig(filename, dpi=500)


plt.cla()
plt.clf()
# cm_analysis(y_true, y_pred, filename=test_sonuc2 + '-2', etikets=etiket, ymap=None, figsize=(10, 10))

# predict probabilities for test set
yhat_probs = model.predict(x_test, verbose=0)
yhat_probs = yhat_probs[:, 0]

##############################################

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_true, y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_true, y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_true, y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_true, y_pred)
print('F1 score: %f' % f1)
# kappa
kappa = cohen_kappa_score(y_true, y_pred)
print('Cohens kappa: %f' % kappa)

###########

bas = balanced_accuracy_score(y_true, y_pred)
print('Balenced Accuracy: %f' % bas)

aps = average_precision_score(y_true, yhat_probs)
print('average_precision_score: %f' % aps)

mc = matthews_corrcoef(y_true, y_pred)
print('matthews_corrcoef: %f' % mc)

fbs = fbeta_score(y_true, y_pred, beta=0.5)
print('fbeta_score: %f' % fbs)

hl = hamming_loss(y_true, y_pred)
print('hamming_loss: %f' % hl)

js = jaccard_score(y_true, y_pred)
print('jaccard_score: %f' % js)
# log_loss(y_true, y_pred[, eps, normalize, …])

prfs = precision_recall_fscore_support(y_true, y_pred, average='weighted')
print('precision_recall_fscore_support:')
print(prfs)

zol = zero_one_loss(y_true, y_pred)
print('zero_one_loss: %f' % zol)

mse = mean_squared_error(y_true, y_pred)
print('mean_squared_error: %f' % mse)

print(classification_report(y_true, y_pred, target_names=etiket))
##############################################

# P-R Grafik
############################################
precision, recall, thresholds = precision_recall_curve(y_true, yhat_probs, pos_label=0)
# print('Precision_recall_curve: %f' % prc)
plt.cla()
plt.clf()
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([-0.01, 1.01])
plt.xlim([-0.01, 1.01])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(aps))
plt.savefig(PR, dpi=500)
##############################################


##############################################
# Plot the roc curve
fpr, tpr, thresholds = roc_curve(y_true, yhat_probs, pos_label=0)
auc = auc(fpr, tpr)
# ROC AUC
# auc = roc_auc_score(y_true, yhat_probs,pos_label=0)
print('ROC AUC: %f' % auc)

plt.cla()
plt.clf()
# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(roc, dpi=500)
##############################################
model.summary()
# cm = ConfusionMatrix(y_true, y_pred)
# cm.print_stats()
