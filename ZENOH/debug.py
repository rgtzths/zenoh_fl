#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Gabriele Baldoni'
__version__ = '0.1'
__email__ = 'gabriele@zettascale.tech'
__status__ = 'Development'

import json
import pathlib
import time
import logging
import numpy as np
import tensorflow as tf
from ZENOH.zcomm import ZComm, ALL_SRC, ANY_SRC, ANY_TAG
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG)

tf.keras.utils.set_random_seed(42)


def run(
    dataset_util, 
    optimizer,
    early_stop,
    learning_rate,
    batch_size, 
    global_epochs, 
    local_epochs, 
    alpha,
    patience,
    min_delta,
    n_workers,
    rank
):

   
    comm = ZComm(rank, n_workers)

    logging.info(f'[RANK: {rank}] Waiting nodes...')
    comm.wait(n_workers+1)
    logging.info(f'[RANK: {rank}] Nodes up!')

   
    if rank == 0:
            for i in range(0,20):
                comm.send(data=i, dest=1, tag=123)
                # comm.send(data=i+1, dest=1, tag=123)
                # time.sleep(1)

        
    else:
        flag = True
        while flag:
            data = comm.recv(source=0, tag=123)
            for (s, t), value in data.items():
                logging.info(f'Sender: {s} - Tag: {t}, Data={value} - type(value): {type(value)}')
                if value == 19:
                    flag = False
   
    comm.close()