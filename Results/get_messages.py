#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import argparse
import json
import pathlib

import numpy as np
from matplotlib import pyplot as plt


def plot_model_history(input_folder, file_name, centralized, size_model, n_batches, n_workers):

    output = pathlib.Path(input_folder)

    model_history = json.load(open(output/ file_name))

    values = list(np.arange(0.50,1,0.05))
    print("Communication round", "MCC", "State", "Communication overhead")
    state = 0
    for i in range(len(model_history["mcc"])):
        if model_history["mcc"][i] >= values[state]:
            while state < len(values) and model_history["mcc"][i] >= values[state]:
                state += 1
            if centralized:
                print(i+1, model_history["mcc"][i], values[state -1], round(size_model*n_batches*n_workers*2*(i+1)*0.000001,2))
            else:
                print(i+1, model_history["mcc"][i], values[state -1], round(size_model*n_workers*2*(i+1)*0.000001,2))

            if state == len(values):
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the model')
    parser.add_argument('-f', type=str, help='Input/output folder', default='results/')
    parser.add_argument('-s', type=str, help='Server file name', default='server.json')
    parser.add_argument('-c', type=bool, help='Is centralized', default=True)
    args = parser.parse_args()

    plot_model_history(args.f, args.s, False, 54917, 299, 8) 
    # Model size 54917 IoT
    # Model size 6033 Slicing
    # If IOT and 128 batch size n_batch = 299
    # If slicing n_batch=292 e  batches