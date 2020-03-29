#!/usr/bin/env python3

from scipy.ndimage.interpolation import shift
import numpy as np

def lateral_shifting(original, n):
    '''
    Generate list of character images shifted LR from -n to n (inclusive)
    '''
    result = []
    for i in range (-n, n+1):
        result.append(shift(original, (0,i), cval=0.0, mode='constant'))
    return result

def main():
    x_train = np.load('./data/train/Competition_Train_data_8000.npy')
    y_train = np.load('./data/train/Competition_Train_label_8000.npy')
    N = 3

    new_x = []
    new_y = []
    for x_i, y_i in zip(x_train, y_train):
        new_x.extend(lateral_shifting(x_i, N))
        new_y.extend([y_i]*(2*N + 1))

    new_x_train = np.asarray(new_x)
    print(x_train.shape)
    print(new_x_train.shape)
    np.save('./data/train/new_x_train.npy', new_x_train)

    new_y_train = np.asarray(new_y)
    np.save('./data/train/new_y_train.npy', new_y_train)
    print(y_train.shape)
    print(new_y_train.shape)

main()
