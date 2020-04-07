#!/usr/bin/env python3

import argparse
import datetime
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import backend as K
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64, help='The batch size (default=64)')
    parser.add_argument('--epoch', type=int, default=128, help='Number of epochs to train (default=128)')
    parser.add_argument('--split', type=int, default=0.2, help='Training validation split (default=0.2)')
    args = parser.parse_args()

    train_data = np.load('./data/train/Competition_Train_data_8000.npy')
    label_data = np.load('./data/train/Competition_Train_label_8000.npy')
    x_train, x_test, y_train, y_test = train_test_split(train_data, label_data, test_size=args.split)

    img_rows = train_data.shape[1] #64
    img_cols = train_data.shape[2] #64
    num_classes = 16
    batch_size = args.batch
    epochs = args.epoch

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    '''
    Started from:
    https://keras.io/examples/mnist_cnn/
    Removing the below made a huge improvement:
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    Additional tweaks inspired from:
    https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/

    We bought a zoo!:
    https://www.asimovinstitute.org/neural-network-zoo/
    '''
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

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
    model.add(Conv2D(32, (3, 3), activation='relu', padding='Same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='Same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='Same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='Same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu', padding='Same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    #RMSprop, Adadelta, Nadam seem best
    optimizer = keras.optimizers.RMSprop()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                               patience=3,
                                               verbose=1,
                                               min_delta=0.0005)

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=args.batch),
                        epochs=args.epoch,
                        steps_per_epoch=(x_train.shape[0]//args.batch)*10,
                        validation_data=(x_test, y_test),
                        callbacks=[learning_rate_reduction, early_stop],
                        verbose=1)

    model.save(f'models/model-b{args.batch}-e{args.epoch}-s{args.split}-{datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S")}')

main()
