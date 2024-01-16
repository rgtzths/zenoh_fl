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
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from ZENOH.zcomm import ZComm, ALL_SRC, ANY_SRC

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG)

tf.keras.utils.set_random_seed(42)

def run(
    dataset_util, 
    optimizer,
    early_stop,
    learning_rate,
    batch_size,
    epochs,
    patience,
    min_delta,
    n_workers,
    rank
):

    stop = False
    dataset = dataset_util.name
    patience_buffer = [0]*patience

    output = f"{dataset}/zenoh/centralized_sync/{n_workers}_{epochs}"
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = pathlib.Path(dataset)

    model = dataset_util.create_model()
    optimizer = optimizer(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    start = time.time()
    comm = ZComm(rank, n_workers)
    logging.info(f'[RANK: {rank}] Epochs: {epochs}')
    logging.info(f'[RANK: {rank}] Waiting nodes...')
    comm.wait(n_workers+1)
    logging.info(f'[RANK: {rank}] Nodes up!')

    if rank == 0:
        node_weights = [0]*n_workers
        X_cv, y_cv = dataset_util.load_validation_data()
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_cv, y_cv)).batch(batch_size)

        #Get the amount of training examples of each worker and divides it by the total
        #of examples to create a weighted average of the model weights
        # for node in range(n_workers):
            
            # n_examples = comm.recv(source=MPI.ANY_SOURCE, tag=1000, status=status)
            # node_weights[status.Get_source()-1] = n_examples
        data = comm.recv(source=ALL_SRC, tag=1000)
        for (k, _tag), v in data.items():
            node_weights[k-1] = v
        # print(f'[RANK: {rank}] node_weights: {node_weights}!')
        
        
        total_size = sum(node_weights)

        node_weights = [weight/total_size for weight in node_weights]
        results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "loads" : []}}
        results["times"]["loads"].append(time.time() - start)
        # print(f'[RANK: {rank}] Results: {results}!')
        total_n_batches = max(node_weights)

    else:
        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)

        train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))

        # comm.send(len(X_train), dest=0, tag=1000)
        comm.send(dest=0, data=len(X_train), tag=1000)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)
    # print(f'[RANK: {rank}] After optimizer!')
    # model.set_weights(comm.bcast(model.get_weights(), root=0))
    model.set_weights(comm.bcast(data=model.get_weights(), root=0, tag=0))
    # print(f'[RANK: {rank}] After bcast!')

    if rank == 0:
        results["times"]["loads"].append(time.time() - start)

    batch = 0
    for epoch in range(epochs):
        weights = []
        if rank == 0:
            avg_grads = []
            #for node in range(n_workers):

            grads_recv = comm.recv(source=ALL_SRC, tag=epoch)
            for (source, _tag), grads in grads_recv.items(): 
                # logging.debug(f'[RANK: {rank}] Data from {source, _tag}')
                if not avg_grads:
                    avg_grads = [grad*node_weights[source-1] for grad in grads]
                else:
                    avg_grads = [ avg_grads[i] + grads[i]*node_weights[source-1] for i in range(len(grads))]

                # grads = comm.recv(source=MPI.ANY_SOURCE, tag=epoch, status=status)
                # source = status.Get_source()

            optimizer.apply_gradients(zip(avg_grads, model.trainable_weights))
            weights = model.get_weights()
            
        else:
            x_batch_train, y_batch_train = train_dataset[batch]

            with tf.GradientTape() as tape:

                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)

            #comm.send(grads, dest=0, tag=epoch)
            comm.send(data=grads, tag=epoch, dest=0)
            # logging.debug(f'[RANK: {rank}] sent to {(0, epoch)}')
            batch = (batch + 1) % len(train_dataset)

        model.set_weights(comm.bcast(data=weights, root=0, tag=0))

        stop = comm.bcast(data=stop, root=0, tag=0)

        if stop:
            break

        # logging.debug(f'[RANK: {rank}] Done epoch {epoch}')
        if (batch+1) % total_n_batches == 0:
            if rank == 0:

                logging.info(f"\n End of batch {batch+1} -> epoch {(batch+1) // total_n_batches}")
                predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
                val_f1 = f1_score(y_cv, predictions, average="macro")
                val_mcc = matthews_corrcoef(y_cv, predictions)
                val_acc = accuracy_score(y_cv, predictions)

                results["acc"].append(val_acc)
                results["f1"].append(val_f1)
                results["mcc"].append(val_mcc)
                results["times"]["global_times"].append(time.time() - start)
                logging.info("- val_f1: %f - val_mcc %f - val_acc %f" %(val_f1, val_mcc, val_acc))
                patience_buffer = patience_buffer[1:]
                patience_buffer.append(val_mcc)
                if val_mcc > early_stop or abs(patience_buffer[0] - patience_buffer[-1]) < min_delta :
                    stop = True

            else:
                results["times"]["epochs"].append(time.time() - epoch_start)
            
            epoch_start = time.time()

    history = json.dumps(results)
    if rank==0:
        model.save(output/'trained_model.keras')
        with open(output/"server.json", "w") as f:
            f.write(history)

    else:
        with open(output/f"worker{rank}.json", "w") as f:
            f.write(history)

    comm.close()
