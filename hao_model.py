#!/usr/bin/env python3

import argparse
import datetime
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Model
import numpy as np

import cv2
import itertools
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D,BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64, help='The batch size (default=64)')
    parser.add_argument('--epoch', type=int, default=128, help='Number of epochs to train (default=128)')
    parser.add_argument('--split', type=int, default=0.2, help='Training validation split (default=0.2)')
    args = parser.parse_args()
    Train_data = np.load('./data/train/Competition_Train_data_8000.npy')
    Train_label = np.load('./data/train/Competition_Train_label_8000.npy')



    x_train, x_test, y_train, y_test = train_test_split(Train_data,
                                                        Train_label,
                                                        test_size=args.split)

    x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[1], x_train.shape[2], 1])
    x_test = np.reshape(x_test, [x_test.shape[0], x_test.shape[1], x_test.shape[2], 1])

    x = np.reshape(Train_data, [Train_data.shape[0], Train_data.shape[1], Train_data.shape[2], 1])

    print('Training set: {} and Training Targets: {}'.format(x_train.shape, y_train.shape))
    print('Test set: {} and test targets: {}'.format(x_test.shape, y_test.shape))

    ## Generate one-hot-vector labels
    y_train_onehot = keras.utils.to_categorical(y_train)
    y_test_onehot = keras.utils.to_categorical(y_test)
    y_onehot = keras.utils.to_categorical(Train_label)

    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (7,7),padding = 'Same',
                    activation ='relu', input_shape = (64,64,1)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation = "softmax"))

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=3,
                                                verbose=0,
                                                factor=0.5,
                                                min_lr=0.00001)

    epochs = args.epoch

    history = model.fit_generator(datagen.flow(x_train,y_train_onehot, batch_size=args.batch),
        epochs = epochs, steps_per_epoch = x.shape[0]//args.batch,
        validation_data = (x_test,y_test_onehot), callbacks=[learning_rate_reduction], verbose=1)

    model.save(f'models/model-b{args.batch}-e{args.epoch}-s{args.split}-{datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S")}')

main()
