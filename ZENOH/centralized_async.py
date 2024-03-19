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


    output = f"{output}/{dataset}/zenoh/centralized_async/{n_workers}_{epochs}_{batch_size}"
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = pathlib.Path(dataset)

    model = dataset_util.create_model()
    optimizer = optimizer(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    start = time.time()
    comm = ZComm(rank, n_workers)


    logging.info(f'[RANK: {rank}] Waiting nodes...')
    comm.wait(n_workers+1)
    logging.info(f'[RANK: {rank}] Nodes up!')
    model_weights = None

    if rank == 0:
        results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "global_times" : []}}
        node_weights = [0]*n_workers

        X_cv, y_cv = dataset_util.load_validation_data()

        val_dataset = tf.data.Dataset.from_tensor_slices((X_cv, y_cv)).batch(batch_size)

        #Get the amount of training examples of each worker and divides it by the total
        #of examples to create a weighted average of the model weights
        data = comm.recv(source=ALL_SRC, tag=1000)
        for (source,src_tag), v in data.items():
            node_weights[source-1] = v
        
        total_n_batches = sum(node_weights)
        total_batches = epochs * total_n_batches

        node_weights = [weight/total_n_batches for weight in node_weights]
        model_weights = model.get_weights()
    else:
        results = {"times" : {"train" : [], "epochs" : []}}

        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)

        train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))

        comm.send(dest=0, data=len(train_dataset), tag=1000)

        total_batches = epochs * len(train_dataset)

    model_weights = comm.bcast(data=model_weights, root=0, tag=0)

    if rank != 0:
        model.set_weights(model_weights)

    epoch_start = time.time()
    if rank == 0:

        exited_workers = 0
        latest_tag = 0

        for batch in range(total_batches):

            data = comm.recv(source=ANY_SRC, tag=ANY_TAG)

            for (source, src_tag), grads in data.items():

                if latest_tag < src_tag+1:
                    latest_tag = src_tag+1

                behind_penalty = (src_tag+1 / latest_tag) 

                grads = [grad*node_weights[source-1]*behind_penalty for grad in grads] 

                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            comm.send(data=model.get_weights(), dest=source, tag=src_tag)

            comm.send(data=stop, dest=source, tag=src_tag)

            if stop:
                exited_workers +=1

            if exited_workers == n_workers:
                break

            if (batch+1) % total_n_batches == 0 and not stop:
                results["times"]["epochs"].append(time.time() - epoch_start)

                logging.info(f"\n End of batch {(batch+1)//n_workers} -> epoch {(batch+1)//total_n_batches}, elapsed time {time.time() - start:.1f}s")

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

                epoch_start = time.time()
                        
    else:
        for batch in range(total_batches):
            train_time = time.time()

            x_batch_train, y_batch_train = train_dataset[batch % len(train_dataset)]

            with tf.GradientTape() as tape:

                logits = model(x_batch_train, training=True) 
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            results["times"]["train"].append(time.time() - train_time)

            comm.send(data=grads, dest=0, tag=batch)

            data = comm.recv(source=0, tag=batch)
            for (s, t), v in data.items():
                model.set_weights(v)
            
            data = comm.recv(source=0, tag=batch)
            for (s, t), v in data.items():
                stop = v

            if (batch+1) % len(train_dataset) == 0:
                results["times"]["epochs"].append(time.time() - epoch_start)
                epoch_start = time.time()

            if stop:
                break
            
    history = json.dumps(results)
    if rank==0:
        logging.info(f'Saving results in {output}')
        with open(output/"server.json", "w") as f:
            f.write(history)
    else:
        with open(output/f"worker{rank}.json", "w") as f:
            f.write(history)

    comm.close()