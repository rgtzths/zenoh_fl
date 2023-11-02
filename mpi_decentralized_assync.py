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

def create_MLP(learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(60,)))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_workers = comm.Get_size()-1
status = MPI.Status()


parser = argparse.ArgumentParser(description='Train and test the model')
parser.add_argument('--g_epochs', type=int, help='Global epochs number', default=10)
parser.add_argument('--l_epochs', type=int, help='local epochs number', default=5)
parser.add_argument('-b', type=int, help='Batch size', default=64)
parser.add_argument('-l', type=float, help='Learning rate', default=0.00001)
parser.add_argument('-d', type=str, help='Dataset', default="one_hot/")
parser.add_argument('-o', type=str, help='Output folder', default="results")
parser.add_argument('-s', type=int, help='Seed for the run', default=42)

args = parser.parse_args()

global_epochs = args.g_epochs
local_epochs = args.l_epochs
batch_size = args.b
learning_rate = args.l
dataset = args.d
output = args.o
tf.keras.utils.set_random_seed(args.s)


output = pathlib.Path(output)
output.mkdir(parents=True, exist_ok=True)
dataset = pathlib.Path(dataset)

model = create_MLP(learning_rate)

start = time.time()

'''
Initial configuration, the parameter server receives the amount of 
examples each worker has to perform a
weighted average of their contributions.
'''

if rank == 0:
    node_weights = [0]*(n_workers)
    X_cv = np.loadtxt(dataset/"x_cv.csv", delimiter=",", dtype=int)
    y_cv = np.loadtxt(dataset/"y_cv.csv", delimiter=",", dtype=int)

    val_dataset = tf.data.Dataset.from_tensor_slices(X_cv).batch(batch_size)

    #Get the amount of training examples of each worker and divides it by the total
    #of examples to create a weighted average of the model weights
    for node in range(n_workers):
        status = MPI.Status()
        n_examples = comm.recv(source=MPI.ANY_SOURCE, tag=1000, status=status)
        node_weights[status.Get_source()-1] = n_examples
    
    total_n_examples = sum(node_weights)

    node_weights = [n_examples/total_n_examples for n_examples in node_weights]
    results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "loads" : []}}
    results["times"]["loads"].append(time.time() - start)

else:
    X_train = np.loadtxt(dataset/("x_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = np.loadtxt(dataset/("y_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = tf.keras.utils.to_categorical(y_train)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

    comm.send(len(X_train), dest=0, tag=1000)

'''
Parameter server shares its values so every worker starts from the same point.
'''
model.set_weights(comm.bcast(model.get_weights(), root=0))


if rank == 0:
    results["times"]["loads"].append(time.time() - start)

'''
Training starts.
Rank 0 is the responsible for aggregating the weights of the models
The remaining perform training
'''
if rank == 0:
    local_weights = model.get_weights()
    for epoch in range(global_epochs*(n_workers)):
        if epoch % n_workers == 0:

            print("\nStart of epoch %d" % (epoch//3))
        
        #This needs to be changed to the correct formula
        weights = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status) 
        source = status.Get_source()

        local_weights = [local_weights[idx] + (((weight - local_weights[idx]) /2) * node_weights[source-1])
                         for idx, weight in enumerate(weights)]

        comm.send(local_weights, dest=source,tag=status.Get_tag())

        if epoch % n_workers == n_workers-1:
            model.set_weights(local_weights)
            predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
            train_f1 = f1_score(y_cv, predictions, average="macro")
            train_mcc = matthews_corrcoef(y_cv, predictions)
            train_acc = accuracy_score(y_cv, predictions)

            results["acc"].append(train_acc)
            results["f1"].append(train_f1)
            results["mcc"].append(train_mcc)
            results["times"]["epochs"].append(time.time() - start)
            print("- val_f1: %f - val_mcc %f - val_acc %f" %(train_f1, train_mcc, train_acc))
        
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()
    
else:
    for global_epoch in range(global_epochs):
        model.fit(train_dataset, epochs=local_epochs, verbose=0)
        
        comm.send(model.get_weights(), dest=0, tag=global_epoch)
        model.set_weights(comm.recv(source=0, tag=global_epoch))    
        
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

if rank==0:
    history = json.dumps(results)

    f = open(output/"train_history.json", "w")
    f.write(history)
    f.close()

    model.save(output/'trained_model.h5')