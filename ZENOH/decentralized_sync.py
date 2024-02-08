#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import json
import pathlib
import time
import logging
import numpy as np
import tensorflow as tf
from ZENOH.zcomm import ZComm, ALL_SRC, ANY_SRC
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
    patience,
    min_delta,
    n_workers,
    rank
):

    stop = False
    dataset = dataset_util.name
    patience_buffer = [-1]*patience

    output = f"/results/{dataset}/zenoh/decentralized_sync/{n_workers}_{global_epochs}_{local_epochs}"
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

    if rank == 0:
        results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "sync" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "global_times" : []}}
        node_weights = [0]*(n_workers)
        X_cv, y_cv = dataset_util.load_validation_data()

        val_dataset = tf.data.Dataset.from_tensor_slices(X_cv).batch(batch_size)

        #Get the amount of training examples of each worker and divides it by the total
        #of examples to create a weighted average of the model weights
        data = comm.recv(source=ALL_SRC, tag=1000)
        for (src, t), nsamples in data.items():
            node_weights[src-1] = nsamples
        
        total_size = sum(node_weights)

        node_weights = [weight/total_size for weight in node_weights]
        results["times"]["sync"].append(time.time() - start)

    else:
        results = {"times" : {"train" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "epochs" : []}}
        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)        

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        comm.send(dest=0, tag=1000, data=len(train_dataset))

    model.set_weights(comm.bcast(data=model.get_weights(), root=0, tag=-10))

    if rank == 0:
        results["times"]["sync"].append(time.time() - start)

    for global_epoch in range(global_epochs):
        epoch_start = time.time()

        avg_weights = []

        if rank == 0:

            logging.info("Start of epoch %d" % global_epoch)
            data = comm.recv(source=ALL_SRC, tag=global_epoch)
            for (source, t), weights in data.items():
                if not avg_weights:
                    avg_weights = [ weight * node_weights[source-1] for weight in weights]
                else:
                    avg_weights = [ avg_weights[i] + weights[i] * node_weights[source-1] for i in range(len(weights))]
            
        else:
            train_time = time.time()
            model.fit(train_dataset, epochs=local_epochs, verbose=0)
            results["times"]["train"].append(time.time() - train_time)
            comm.send(dest=0, tag=global_epoch, data=model.get_weights())

        model.set_weights(comm.bcast(data=avg_weights, root=0, tag=-10))

        if rank != 0:
            results["times"]["epochs"].append(time.time() - epoch_start)

        stop = comm.bcast(data=stop, root=0, tag=-10)

        if stop:
            break

        if rank == 0:
            
            results["times"]["epochs"].append(time.time() - start)

            predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
            val_f1 = f1_score(y_cv, predictions, average="macro")
            val_mcc = matthews_corrcoef(y_cv, predictions)
            val_acc = accuracy_score(y_cv, predictions)

            results["acc"].append(val_acc)
            results["f1"].append(val_f1)
            results["mcc"].append(val_mcc)
            results["times"]["global_times"].append(time.time() - start)

            patience_buffer = patience_buffer[1:]
            patience_buffer.append(val_mcc)
            logging.info("- val_f1: %6.3f - val_mcc %6.3f - val_acc %6.3f" %(val_f1, val_mcc, val_acc))
            
            p_stop = True
            for value in patience_buffer[1:]:
                if abs(patience_buffer[0] - value) > min_delta:
                    p_stop = False 

            if (val_mcc > early_stop or p_stop) and global_epoch > 10:
                stop = True

    history = json.dumps(results)
    if rank==0:
        logging.info(f'Saving results in {output}')
        model.save(output/'trained_model.keras')
        with open(output/"server.json", "w") as f:
            f.write(history)
    else:
        with open(output/f"worker{rank}.json", "w") as f:
            f.write(history)

    comm.close()