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

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def create_MLP():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(60,)))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))

    return model

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_workers = comm.Get_size()-1
status = MPI.Status()

parser = argparse.ArgumentParser(description='Train and test the model')
parser.add_argument('-e', type=int, help='Epochs number', default=1000)
parser.add_argument('-l', type=float, help='Learning rate', default=0.00001)
parser.add_argument('-d', type=str, help='Dataset', default="one_hot/")
parser.add_argument('-o', type=str, help='Output folder', default="results")
parser.add_argument('-b', type=int, help='Batch size', default=1024)
parser.add_argument('-s', type=int, help='Seed for the run', default=42)

args = parser.parse_args()

epochs = args.e
learning_rate = args.l
dataset = args.d
output = args.o
batch_size = args.b
tf.keras.utils.set_random_seed(args.s)

output = pathlib.Path(output)
output.mkdir(parents=True, exist_ok=True)
dataset = pathlib.Path(dataset)

model = create_MLP()
start = time.time()
buff = bytearray(262144)
pickle =  MPI.Pickle()


if rank == 0:
    results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "sync" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "global_times" : []}}
    node_weights = [0]*n_workers
    X_cv = np.loadtxt(dataset/"x_cv.csv", delimiter=",", dtype=int)
    y_cv = np.loadtxt(dataset/"y_cv.csv", delimiter=",", dtype=int)

    val_dataset = tf.data.Dataset.from_tensor_slices(X_cv).batch(batch_size)

    #Get the amount of training examples of each worker and divides it by the total
    #of examples to create a weighted average of the model weights
    for node in range(n_workers):

        comm.Recv(buff, source=MPI.ANY_SOURCE, tag=1000, status=status)

        n_examples = pickle.loads(buff)
        
        node_weights[status.Get_source()-1] = n_examples
    
    total_size = sum(node_weights)
    n_batches = (max(node_weights) // batch_size)+1

    node_weights = [weight/total_size for weight in node_weights]
    results["times"]["sync"].append(time.time() - start)

else:
    results = {"times" : {"train" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "epochs" : []}}

    X_train = np.loadtxt(dataset/("x_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = np.loadtxt(dataset/("y_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = tf.keras.utils.to_categorical(y_train)

    train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    compr_data = pickle.dumps(len(X_train))

    n_batches = (len(X_train) // batch_size)+1

    comm.Send(compr_data, dest=0, tag=1000)

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

weights = bytearray(pickle.dumps(model.get_weights()))

comm.Bcast(weights, root=0)

if rank != 0:
    weights = pickle.loads(weights)

    model.set_weights(weights)

if rank == 0:
    results["times"]["sync"].append(time.time() - start)

batch = 0
epoch_start = time.time()
for epoch in range(epochs):
    weights = []
    if rank == 0:
        avg_grads = []
        for node in range(n_workers):

            com_time = time.time()
            comm.Recv(buff, source=MPI.ANY_SOURCE, tag=epoch, status=status)
            results["times"]["comm_recv"].append(time.time() - com_time)

            load_time = time.time()
            grads = pickle.loads(buff)
            results["times"]["conv_recv"].append(time.time() - load_time)

            source = status.Get_source()

            if not avg_grads:
                avg_grads = [grad*node_weights[source-1] for grad in grads]
            else:
                avg_grads = [ avg_grads[i] + grads[i]*node_weights[source-1] for i in range(len(grads))]
        
        optimizer.apply_gradients(zip(avg_grads, model.trainable_weights))
        load_time = time.time()
        weights = bytearray(pickle.dumps(model.get_weights()))
        results["times"]["conv_send"].append(time.time()- load_time)
        
    else:
        train_time = time.time()

        x_batch_train, y_batch_train = train_dataset[batch]

        with tf.GradientTape() as tape:

            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        results["times"]["train"].append(time.time() - train_time)

        load_time = time.time()
        grads = pickle.dumps(grads)
        results["times"]["conv_send"].append(time.time() - load_time)
        
        comm_time = time.time()
        comm.Send(grads, dest=0, tag=epoch)
        results["times"]["comm_send"].append(time.time() - comm_time)

        batch = (batch + 1) % len(train_dataset)
        weights = buff

    com_time = time.time()
    comm.Bcast(weights, root=0)

    if rank == 0:
        results["times"]["comm_send"].append(time.time() - com_time)
    else:
        results["times"]["comm_recv"].append(time.time() - com_time)

        conv_time = time.time()
        weights = pickle.loads(weights)
        results["times"]["conv_recv"].append(time.time() - conv_time)

        model.set_weights(weights)

    if epoch % n_batches == 0 or epoch == epochs-1:
        if rank == 0:
            results["times"]["epochs"].append(time.time() - epoch_start)

            print(f"\n End of batch {epoch} -> epoch {epoch // n_batches}")
            predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
            train_f1 = f1_score(y_cv, predictions, average="macro")
            train_mcc = matthews_corrcoef(y_cv, predictions)
            train_acc = accuracy_score(y_cv, predictions)

            results["acc"].append(train_acc)
            results["f1"].append(train_f1)
            results["mcc"].append(train_mcc)
            results["times"]["global_times"].append(time.time() - start)
            print("- val_f1: %f - val_mcc %f - val_acc %f" %(train_f1, train_mcc, train_acc))

        else:
            results["times"]["epochs"].append(time.time() - epoch_start)

        epoch_start = time.time()

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()

history = json.dumps(results)
if rank==0:
    results_dir = output/f"parameter_server"
    results_dir.mkdir(parents=True, exist_ok=True)

else:
    results_dir = output/f"worker{rank}"
    results_dir.mkdir(parents=True, exist_ok=True)

f = open(results_dir/"train_history.json", "w")
f.write(history)
f.close()

model.save(results_dir/'trained_model.keras')