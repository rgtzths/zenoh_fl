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
import random
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
    stop = False
    dataset = dataset_util.name
    patience_buffer = [0]*patience

    output = f"{dataset}/zenoh/decentralized_async/{n_workers}_{global_epochs}_{local_epochs}_{alpha}"
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = pathlib.Path(dataset)
    
    model = dataset_util.create_model()
    model.compile(
        optimizer=optimizer(learning_rate=learning_rate), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    start = time.time()
    comm = ZComm(rank, n_workers)

    logging.info(f'[RANK: {rank}] Waiting nodes...')
    comm.wait(n_workers+1)
    logging.info(f'[RANK: {rank}] Nodes up!')

    '''
    Initial configuration, the parameter server receives the amount of 
    examples each worker has to perform a
    weighted average of their contributions.
    '''

    if rank == 0:
        results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "sync" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "global_times" : []}}

        node_weights = [0]*(n_workers)
        X_cv, y_cv = dataset_util.load_validation_data()

        val_dataset = tf.data.Dataset.from_tensor_slices(X_cv).batch(batch_size)

        #Get the amount of training examples of each worker and divides it by the total
        #of examples to create a weighted average of the model weights
        data = comm.recv(source=ALL_SRC, tag=1000)
        for (source,src_tag), n_examples in data.items():
            node_weights[source-1] = n_examples
        
        biggest_n_examples = max(node_weights)

        node_weights = [n_examples/biggest_n_examples for n_examples in node_weights]

        results["times"]["sync"].append(time.time() - start)

    else:
        results = {"times" : {"train" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "epochs" : []}}

        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

        comm.send(dest=0, data=len(train_dataset), tag=1000)

    '''
    Parameter server shares its values so every worker starts from the same point.
    '''
    model.set_weights(comm.bcast(data=model.get_weights(), root=0, tag=0))

    if rank == 0:
        results["times"]["sync"].append(time.time() - start)
   

    logging.info(f'[RANK: {rank}] Starting async comms!')

    if rank == 0:
        r = 0 
        while r<80:
            data = comm.recv(source=ANY_SRC, tag=ANY_TAG)
            for (s, t), value in data.items():
                logging.info(f'[RANK: {rank}] Received = Sender: {s} - Tag: {t}, Data={value}')
                r+=1

        for i in range(0,20):
            for d in range(1, n_workers+1):
                comm.send(data=i, dest=d, tag=123)
                # comm.send(data=i+1, dest=1, tag=123)
                # time.sleep(1)
    else:
        for i in range(0, 20):
            comm.send(data=i, dest=0, tag=(10*rank))
            time.sleep(random.random())
        time.sleep(5)
        flag = True
        while flag:
            data = comm.recv(source=0, tag=123)
            for (s, t), value in data.items():
                logging.info(f'[RANK {rank}] Received = Sender: {s} - Tag: {t}, Data={value}')
                if value == 19:
                    flag = False
        

    logging.info(f'[RANK: {rank}] Done')
    comm.close()