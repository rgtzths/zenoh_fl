#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import argparse
from scipy.stats import f_oneway
import os
import json
import numpy as np

parser = argparse.ArgumentParser(description='Train and test the model')
parser.add_argument('-f', type=str, help='Input/output folder', default='results/')
parser.add_argument('-s', type=str, help='Server file name', default='server.json')

args = parser.parse_args()
threshold_values = list(np.arange(0.50,1,0.1)) + [0.99, 1]
metric = "mcc"

datasets = [f.path for f in os.scandir(args.f) if f.is_dir()]
final_statistics = {}
for dataset in datasets:
    dataset_name = dataset.split("/")[-1]
    if dataset_name not in final_statistics:
        final_statistics[dataset_name] = {}
    seeds = [f.path for f in os.scandir(dataset) if f.is_dir()]
    for seed in seeds:
        seed_value = seed.split("/")[-1]
        comms = [f.path for f in os.scandir(seed) if f.is_dir()]
        for comm in comms:
            comm_name = comm.split("/")[-1]
            if comm_name not in final_statistics[dataset_name]:
                final_statistics[dataset_name][comm_name] = {} 
            fl_approaches = [f.path for f in os.scandir(comm) if f.is_dir()]
            for fl_approach in fl_approaches:
                fl_name = fl_approach.split("/")[-1]
                if comm_name not in final_statistics[dataset_name][comm_name]:
                    final_statistics[dataset_name][comm_name][fl_name] = {} 

                specific_run = [f.path for f in os.scandir(fl_approach) if f.is_dir()][0]
                results_file = json.load(open(f"{specific_run}/server.json", "r"))
                state = 0
                points_interest = []
                for i in range(len(results_file[metric])):
                    if results_file[metric][i] >= threshold_values[state]:
                        while state < len(threshold_values) and results_file[metric][i] >= threshold_values[state]:
                            state += 1

                        if threshold_values[state-1] in final_statistics[dataset_name][comm_name][fl_name]:
                            final_statistics[dataset_name][comm_name][fl_name][threshold_values[state-1]].append(results_file["times"]["global_times"][i])
                        else:
                            final_statistics[dataset_name][comm_name][fl_name][threshold_values[state-1]] = [results_file["times"]["global_times"][i]]

                        if state == len(threshold_values):
                            break
#print(final_statistics)
for dataset in final_statistics:
    print(f"-------{dataset}--------")
    for comm in final_statistics[dataset]:
        print(f"---------{comm}---------")
        for fl_approach in final_statistics[dataset][comm]:
            print(f"-------------{fl_approach}-----------")
            for threshold in final_statistics[dataset][comm][fl_approach]:
                average_value = sum(final_statistics[dataset][comm][fl_approach][threshold]) / len(final_statistics[dataset][comm][fl_approach][threshold])
                print(f"{threshold:.2f} {average_value:.1f}")
            
