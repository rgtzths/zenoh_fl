#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Gabriele Baldoni'
__version__ = '0.1'
__email__ = 'gabriele@zettascale.tech'
__status__ = 'Development'


import argparse
import gc
import json
import pathlib
import time
import logging
import numpy as np
import tensorflow as tf
from zcomm import ZComm, ALL_SRC, ANY_SRC, ANY_TAG
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

def create_MLP():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(60,)))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))

    return model

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Train and test the model')
parser.add_argument('-e', type=int, help='Epochs number', default=100000)
parser.add_argument('-l', type=float, help='Learning rate', default=0.00001)
parser.add_argument('-d', type=str, help='Dataset', default="one_hot/")
parser.add_argument('-o', type=str, help='Output folder', default="results")
parser.add_argument('-b', type=int, help='Batch size', default=64)
parser.add_argument('-s', type=int, help='Seed for the run', default=42)
parser.add_argument('-w', type=int, help='workers', default=3)
parser.add_argument('-r', type=int, help='rank')

args = parser.parse_args()


epochs = args.e
learning_rate = args.l
dataset = args.d
output = args.o
batch_size = args.b
n_workers = args.w
rank = args.r
tf.keras.utils.set_random_seed(args.s)

output = pathlib.Path(output)
output.mkdir(parents=True, exist_ok=True)
dataset = pathlib.Path(dataset)

model = create_MLP()
start = time.time()
comm = ZComm(rank, n_workers)


logging.info(f'[RANK: {rank}] Waiting nodes...')
comm.wait(n_workers+1)
logging.info(f'[RANK: {rank}] Nodes up!')

if rank == 0:
    node_weights = [0]*n_workers
    X_cv = np.loadtxt(dataset/"x_cv.csv", delimiter=",", dtype=int)
    y_cv = np.loadtxt(dataset/"y_cv.csv", delimiter=",", dtype=int)

    val_dataset = tf.data.Dataset.from_tensor_slices(X_cv).batch(batch_size)

    #Get the amount of training examples of each worker and divides it by the total
    #of examples to create a weighted average of the model weights
    # for node in range(n_workers):
    #     n_examples = comm.recv(source=MPI.ANY_SOURCE, tag=1000, status=status)
    #     node_weights[status.Get_source()-1] = n_examples
    data = comm.recv(source=ALL_SRC, tag=1000)
    for (k,_), v in data.items():
        node_weights[k-1] = v
    
    total_n_examples = sum(node_weights)

    node_weights = [weight/total_n_examples for weight in node_weights]
    results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "loads" : []}}
    results["times"]["loads"].append(time.time() - start)

else:
    X_train = np.loadtxt(dataset/("x_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = np.loadtxt(dataset/("y_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = tf.keras.utils.to_categorical(y_train)

    train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    #comm.send(len(X_train), dest=0, tag=1000)
    comm.send(dest=0, data=len(X_train), tag=1000)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)

# model.set_weights(comm.bcast(model.get_weights(), root=0))
model.set_weights(comm.bcast(data=model.get_weights(), root=0, tag=0))


if rank == 0:
    results["times"]["loads"].append(time.time() - start)

if rank == 0:
    logging.debug(f'[RANK: {rank}] Expected loops {epochs*n_workers}')
    for batch in range(epochs*n_workers):
        logging.debug(f'[RANK: {rank}] Batch {batch}')
        data = comm.recv(source=ANY_SRC, tag=ANY_TAG)
        for (source, src_tag), grads in data.items():

        # grads = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        # source = status.Get_source()
            logging.debug(f'Data from {source, src_tag}')

            grads = [grad*node_weights[source-1] for grad in grads] # Needs to be updated to the correct penalization format

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # comm.send(model.get_weights(), dest=source, tag=status.Get_tag())
            # TODO: return tag in recv
            comm.send(data=model.get_weights(), dest=source, tag=src_tag)
            logging.debug(f'Sent to {source, src_tag}')

            if (batch // n_workers) % 1500 == 0:
                logging.info("End of epoch %d" % ((batch//n_workers)))

                predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
                train_f1 = f1_score(y_cv, predictions, average="macro")
                train_mcc = matthews_corrcoef(y_cv, predictions)
                train_acc = accuracy_score(y_cv, predictions)

                results["acc"].append(train_acc)
                results["f1"].append(train_f1)
                results["mcc"].append(train_mcc)
                results["times"]["epochs"].append(time.time() - start)
                logging.info("- val_f1: %f - val_mcc %f - val_acc %f" %(train_f1, train_mcc, train_acc))

            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            gc.collect()
                    
else:
    batch = 0
    logging.debug(f'[RANK: {rank}] Expected loops {epochs}')
    for epoch in range(epochs):
        logging.debug(f'[RANK: {rank}] epoch {epoch}')
        x_batch_train, y_batch_train = train_dataset[batch]

        with tf.GradientTape() as tape:

            logits = model(x_batch_train, training=True) 
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)

        #comm.send(grads, dest=0, tag=epoch)
        comm.send(data=grads, dest=0, tag=epoch)
        logging.debug(f'[RANK: {rank}] sent to {(0, epoch)}')

        #model.set_weights(comm.recv(source=0, tag=epoch))
        data = comm.recv(source=0, tag=epoch)
        for (s, t), v in data.items():
            logging.debug(f'[RANK: {rank}] recv from {(s, t)}')

            model.set_weights(v)

        batch = (batch + 1) % len(train_dataset)

        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

if rank==0:
    history = json.dumps(results)
    logging.info(f'Saving results in {output}')
    f = open( output/"train_history.json", "w")
    f.write(history)
    f.close()

    model.save(output/'trained_model.h5')

comm.close()