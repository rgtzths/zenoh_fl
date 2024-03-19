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
    rank,
    output
):

    stop = False
    dataset = dataset_util.name
    patience_buffer = [-1]*patience

    output = f"{output}/{dataset}/zenoh/centralized_sync/{n_workers}_{epochs}_{batch_size}"
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = pathlib.Path(dataset)

    model = dataset_util.create_model()
    optimizer = optimizer(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model_weights = None


    start = time.time()
    comm = ZComm(rank, n_workers)
    logging.info(f'[RANK: {rank}] Epochs: {epochs}')
    logging.info(f'[RANK: {rank}] Waiting nodes...')
    comm.wait(n_workers+1)
    logging.info(f'[RANK: {rank}] Nodes up!')

    if rank == 0:
        results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "global_times" : []}}
        node_weights = [0]*n_workers
        X_cv, y_cv = dataset_util.load_validation_data()
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_cv, y_cv)).batch(batch_size)

        #Get the amount of training examples of each worker and divides it by the total
        #of examples to create a weighted average of the model weights
        data = comm.recv(source=ALL_SRC, tag=1000)
        for (k, _tag), v in data.items():
            node_weights[k-1] = v
        
        sum_n_batches = sum(node_weights)
        total_n_batches = max(node_weights)
        total_batches = epochs * total_n_batches

        node_weights = [weight/sum_n_batches for weight in node_weights]
        model_weights = model.get_weights()
    else:
        results = {"times" : {"train" : [], "epochs" : []}}
        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)

        train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))

        total_n_batches = len(train_dataset)
        total_batches = epochs * total_n_batches

        comm.send(dest=0, data=total_n_batches, tag=1000)

    model_weights = comm.bcast(data=model_weights, root=0, tag=0)

    if rank != 0:
        model.set_weights(model_weights)
        
    epoch_start = time.time()
    for batch in range(total_batches):
        weights = []
        if rank == 0:
            avg_grads = []

            grads_recv = comm.recv(source=ALL_SRC, tag=batch)
            for (source, _tag), grads in grads_recv.items(): 

                if not avg_grads:
                    avg_grads = [grad*node_weights[source-1] for grad in grads]
                else:
                    avg_grads = [ avg_grads[i] + grads[i]*node_weights[source-1] for i in range(len(grads))]

            optimizer.apply_gradients(zip(avg_grads, model.trainable_weights))
            model_weights = model.get_weights()
            
        else:
            train_time = time.time()
            x_batch_train, y_batch_train = train_dataset[batch % len(train_dataset)]

            with tf.GradientTape() as tape:

                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)

            results["times"]["train"].append(time.time() - train_time)

            comm.send(data=grads, tag=batch, dest=0)

        model_weights = comm.bcast(data=model_weights, root=0, tag=0)

        stop = comm.bcast(data=stop, root=0, tag=0)

        if stop:
            break
        
        if rank != 0:
            model.set_weights(model_weights)

        if (batch+1) % total_n_batches == 0:

            if rank == 0:

                logging.info(f"\n End of batch {batch+1} -> epoch {(batch+1) // total_n_batches}, elapsed time {time.time() - start:.1f}s")
                predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
                val_f1 = f1_score(y_cv, predictions, average="macro")
                val_mcc = matthews_corrcoef(y_cv, predictions)
                val_acc = accuracy_score(y_cv, predictions)

                results["acc"].append(val_acc)
                results["f1"].append(val_f1)
                results["mcc"].append(val_mcc)
                results["times"]["global_times"].append(time.time() - start)
                logging.info("- val_f1: %6.3f - val_mcc %6.3f - val_acc %6.3f" %(val_f1, val_mcc, val_acc))
                patience_buffer = patience_buffer[1:]
                patience_buffer.append(val_mcc)
                
                p_stop = True
                for value in patience_buffer[1:]:
                    if abs(patience_buffer[0] - value) > min_delta:
                        p_stop = False 

                if val_mcc > early_stop or p_stop:
                    stop = True

            else:
                results["times"]["epochs"].append(time.time() - epoch_start)
            
            epoch_start = time.time()

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
