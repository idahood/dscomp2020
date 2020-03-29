#!/usr/bin/env python3

import argparse
import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32, help='The batch size (default=32)')
    parser.add_argument('--epoch', type=int, default=128, help='Number of epochs to train (default=128)')
    args = parser.parse_args()
    x_train = np.load('./data/train/Competition_Train_data_8000.npy')
    y_train = np.load('./data/train/Competition_Train_label_8000.npy')

    assert(x_train.shape == (8000, 64, 64))
    assert(y_train.shape == (8000,))

    img_rows = 64
    img_cols = 64

    # Hyperparamaters
    num_classes = 16
    batch_size = args.batch
    epochs = args.epoch

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    '''
    And we're stealing from https://keras.io/examples/mnist_cnn/
    '''
    x_train = x_train.astype('float32')
    x_train /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Try Adam as optimizer
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # Enable shuffle=True? No
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.1)

    model.save(f'models/model-{datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S")}')

main()
