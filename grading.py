#!/usr/bin/env python3

import argparse
import time
import numpy as np

from keras.models import load_model

def check(e1, e4, e5):
    # Worth 10 points
    if (e1==0 and e4==7 and e5==11):
        return 10
    elif(e1==1 and e4==8 and e5==11):
        return 10
    elif(e1==2 and e4==9 and e5==11):
        return 10
    elif(e1==3 and e4==10 and e5==11):
        return 10
    elif(e1==4 and e4==1 and e5==11):
        return 10
    elif(e1==5 and e4==2 and e5==11):
        return 10
    elif(e1==6 and e4==3 and e5==11):
        return 10

    # Worth 9 points
    elif(e1==0 and e4==7 and e5!=11):
        return 9
    elif(e1==1 and e4==8 and e5!=11):
        return 9
    elif(e1==2 and e4==9 and e5!=11):
        return 9
    elif(e1==3 and e4==10 and e5!=11):
        return 9
    elif(e1==4 and e4==1 and e5!=11):
        return 9
    elif(e1==5 and e4==2 and e5!=11):
        return 9
    elif(e1==6 and e4==3 and e5!=11):
        return 9

    # Worth 5 points
    elif(e1==0 and e4==12 and e5==11):
        return 5
    elif(e1==1 and e4==13 and e5==11):
        return 5
    elif(e1==2 and e4==14 and e5==11):
        return 5
    elif(e1==3 and e4==15 and e5==11):
        return 5

    # Worth 2 points
    elif(e1==1 and e4==4 and e5==11):
        return 2
    elif(e1==2 and e4==5 and e5==11):
        return 2
    elif(e1==3 and e4==6 and e5==11):
        return 2

    # Worth 1 point
    elif(e1==1 and e4==4 and e5!=11):
        return 1
    elif(e1==2 and e4==5 and e5!=11):
        return 1
    elif(e1==3 and e4==6 and e5!=11):
        return 1

    else:
        return 0

def main():
    '''
    Intergral is always of shape 384x64
    Roughly formed by 6 64x64 characters
    We never care about the integeral symbol, dx, or =
    So for a given list equation some indices are fixed:
    equation[0] = integral symbol
    equation[2] = dx
    equation[3] = '='

    Put another way, we only check e[1],e[4], and e[5]
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='The path to the keras model')
    args = parser.parse_args()

    start = time.perf_counter()

    CHARACTERS = 6
    x_test = np.load('./data/test/Competition_Problems.npy')
    model = load_model(args.path)
    model.summary()

    result = []

    for i, equation in enumerate(x_test):
        characters = np.array_split(equation, CHARACTERS, axis=1)
        evaluated = []
        for char in characters:
            pred = model.predict(char.reshape(1, 64, 64, 1))
            evaluated.append(np.argmax(pred))

        worth = check(evaluated[1], evaluated[4], evaluated[5])

        result.append(f'{i}, {worth}\n')

    with open('./output.csv', 'w') as fh:
        fh.writelines(result)

    end = time.perf_counter()
    print(end-start)

main()
