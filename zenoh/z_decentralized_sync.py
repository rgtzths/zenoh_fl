#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import argparse
import gc
import json
import pathlib
import time
import logging
import numpy as np
import tensorflow as tf
from zcomm import ZComm, ALL_SRC, ANY_SRC
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def create_MLP(learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(60,)))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG)


parser = argparse.ArgumentParser(description='Train and test the model')
parser.add_argument('--g_epochs', type=int, help='Global epochs number', default=10)
parser.add_argument('--l_epochs', type=int, help='local epochs number', default=5)
parser.add_argument('-b', type=int, help='Batch size', default=64)
parser.add_argument('-l', type=float, help='Learning rate', default=0.00001)
parser.add_argument('-d', type=str, help='Dataset', default="one_hot/")
parser.add_argument('-o', type=str, help='Output folder', default="results")
parser.add_argument('-s', type=int, help='Seed for the run', default=42)
parser.add_argument('-w', type=int, help='workers', default=3)
parser.add_argument('-r', type=int, help='rank')

args = parser.parse_args()

global_epochs = args.g_epochs
local_epochs = args.l_epochs
batch_size = args.b
learning_rate = args.l
dataset = args.d
output = args.o
n_workers = args.w
rank = args.r
tf.keras.utils.set_random_seed(args.s)


output = pathlib.Path(output)
output.mkdir(parents=True, exist_ok=True)
dataset = pathlib.Path(dataset)

model = create_MLP(learning_rate)

start = time.time()
comm = ZComm(rank, n_workers)

logging.info(f'[RANK: {rank}] Waiting nodes...')
comm.wait(n_workers+1)
logging.info(f'[RANK: {rank}] Nodes up!')

if rank == 0:
    node_weights = [0]*(n_workers)
    X_cv = np.loadtxt(dataset/"x_cv.csv", delimiter=",", dtype=int)
    y_cv = np.loadtxt(dataset/"y_cv.csv", delimiter=",", dtype=int)

    val_dataset = tf.data.Dataset.from_tensor_slices(X_cv).batch(batch_size)

    #Get the amount of training examples of each worker and divides it by the total
    #of examples to create a weighted average of the model weights
    # for node in range(n_workers):
    #   n_examples = comm.recv(source=MPI.ANY_SOURCE tag=1000, status=status)
    data = comm.recv(source=ALL_SRC, tag=1000)
    for (src, t), nsamples in data.items():
        node_weights[src-1] = nsamples
    
    total_size = sum(node_weights)

    node_weights = [weight/total_size for weight in node_weights]
    results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "loads" : []}}
    results["times"]["loads"].append(time.time() - start)

else:
    X_train = np.loadtxt(dataset/("x_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = np.loadtxt(dataset/("y_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = tf.keras.utils.to_categorical(y_train)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    comm.send(dest=0, tag=1000, data=len(X_train))
    #comm.send(len(X_train), dest=0, tag=1000)

model.set_weights(comm.bcast(data=model.get_weights(), root=0, tag=-10))

if rank == 0:
    results["times"]["loads"].append(time.time() - start)

for global_epoch in range(global_epochs):
    avg_weights = []

    if rank == 0:

        logging.info("\nStart of epoch %d" % global_epoch)
        # for node in range(n_workers):
        #     weights = comm.recv(source=MPI.ANY_SOURCE, tag=global_epoch, status=status)
        #     source = status.Get_source()
        data = comm.recv(source=ALL_SRC, tag=global_epoch)
        for (source, t), weights in data.items():
            if not avg_weights:
                avg_weights = [ weight * node_weights[source-1] for weight in weights]
            else:
                avg_weights = [ avg_weights[i] + weights[i] * node_weights[source-1] for i in range(len(weights))]
        
    else:
        model.fit(train_dataset, epochs=local_epochs, verbose=0)
        #comm.send(model.get_weights(), dest=0, tag=global_epoch)
        comm.send(dest=0, tag=global_epoch, data=model.get_weights())

    model.set_weights(comm.bcast(data=avg_weights, root=0, tag=-10))

    if rank == 0:
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

if rank==0:
    history = json.dumps(results)

    f = open( output/"train_history.json", "w")
    f.write(history)
    f.close()

    model.save(output/'trained_model.h5')

comm.close()