#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import argparse
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
buff = bytearray(262144)
pickle =  MPI.Pickle()

parser = argparse.ArgumentParser(description='Train and test the model')
parser.add_argument('--g_epochs', type=int, help='Global epochs number', default=10)
parser.add_argument('--l_epochs', type=int, help='local epochs number', default=5)
parser.add_argument('-b', type=int, help='Batch size', default=1024)
parser.add_argument('-l', type=float, help='Learning rate', default=0.00001)
parser.add_argument('-d', type=str, help='Dataset', default="one_hot/")
parser.add_argument('-o', type=str, help='Output folder', default="results/decent_async")
parser.add_argument('-s', type=int, help='Seed for the run', default=42)
parser.add_argument('-a', type=float, help='Updated bound for the master/worker', default=0.2)

args = parser.parse_args()

global_epochs = args.g_epochs
local_epochs = args.l_epochs
batch_size = args.b
learning_rate = args.l
dataset = args.d
output = args.o
alpha = args.a
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
    results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "sync" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "global_times" : []}}

    node_weights = [0]*(n_workers)
    X_cv = np.loadtxt(dataset/"x_cv.csv", delimiter=",", dtype=int)
    y_cv = np.loadtxt(dataset/"y_cv.csv", delimiter=",", dtype=int)

    val_dataset = tf.data.Dataset.from_tensor_slices(X_cv).batch(batch_size)

    #Get the amount of training examples of each worker and divides it by the total
    #of examples to create a weighted average of the model weights
    for node in range(n_workers):
        status = MPI.Status()
        comm.Recv(buff, source=MPI.ANY_SOURCE, tag=1000, status=status)
        n_examples = pickle.loads(buff)

        node_weights[status.Get_source()-1] = n_examples
    
    biggest_n_examples = max(node_weights)

    node_weights = [n_examples/biggest_n_examples for n_examples in node_weights]

    results["times"]["sync"].append(time.time() - start)
    weights = bytearray(pickle.dumps(model.get_weights()))

else:
    results = {"times" : {"train" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "epochs" : []}}

    X_train = np.loadtxt(dataset/("x_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = np.loadtxt(dataset/("y_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = tf.keras.utils.to_categorical(y_train)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

    compr_data = pickle.dumps(len(X_train))

    comm.Send(compr_data, dest=0, tag=1000)
    weights = buff


'''
Parameter server shares its values so every worker starts from the same point.
'''
comm.Bcast(weights, root=0)

if rank != 0:
    weights = pickle.loads(weights)

    model.set_weights(weights)
else:
    results["times"]["sync"].append(time.time() - start)

'''
Training starts.
Rank 0 is the responsible for aggregating the weights of the models
The remaining perform training
'''
if rank == 0:
    local_weights = model.get_weights()
    epoch_start = time.time()

    for epoch in range(global_epochs*(n_workers)):
        if epoch % n_workers == 0:

            print("\nStart of epoch %d" % (epoch//n_workers+1))
        
        #This needs to be changed to the correct formula
        
        com_time = time.time()
        comm.Recv(buff, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        results["times"]["comm_recv"].append(time.time() - com_time)

        load_time = time.time()
        weights = pickle.loads(buff)
        results["times"]["conv_recv"].append(time.time() - load_time)

        source = status.Get_source()
        tag = status.Get_tag()

        #Check how to combine here
        weight_diffs = [ (weight - local_weights[idx])*alpha*node_weights[source-1]
                         for idx, weight in enumerate(weights)]
        
        local_weights = [local_weights[idx] + weight
                         for idx, weight in enumerate(weight_diffs)]
        
        load_time = time.time()
        weights = pickle.dumps(weight_diffs)
        results["times"]["conv_send"].append(time.time() - load_time)
        
        comm_time = time.time()
        comm.Send(weights, dest=source, tag=tag)
        results["times"]["comm_send"].append(time.time() - comm_time)

        if epoch % n_workers == n_workers-1:
            results["times"]["epochs"].append(time.time() - epoch_start)
            model.set_weights(local_weights)
            predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
            train_f1 = f1_score(y_cv, predictions, average="macro")
            train_mcc = matthews_corrcoef(y_cv, predictions)
            train_acc = accuracy_score(y_cv, predictions)

            results["acc"].append(train_acc)
            results["f1"].append(train_f1)
            results["mcc"].append(train_mcc)
            results["times"]["global_times"].append(time.time() - start)
            print("- val_f1: %6.3f - val_mcc %6.3f - val_acc %6.3f" %(train_f1, train_mcc, train_acc))
            epoch_start = time.time()

else:
    for global_epoch in range(global_epochs):
        epoch_start = time.time()

        train_time = time.time()
        model.fit(train_dataset, epochs=local_epochs, verbose=0)
        results["times"]["train"].append(time.time() - train_time)

        load_time = time.time()
        weights = pickle.dumps(model.get_weights())
        results["times"]["conv_send"].append(time.time() - load_time)
        
        comm_time = time.time()
        comm.Send(weights, dest=0, tag=global_epoch)
        results["times"]["comm_send"].append(time.time() - comm_time)

        com_time = time.time()
        comm.Recv(buff, source=0, tag=global_epoch)
        results["times"]["comm_recv"].append(time.time() - com_time)

        load_time = time.time()
        weight_diffs = pickle.loads(buff)
        results["times"]["conv_recv"].append(time.time() - load_time)

        weights = [weight - weight_diffs[idx]
                         for idx, weight in enumerate(model.get_weights())]
        
        model.set_weights(weights)    
        
        results["times"]["epochs"].append(time.time() - epoch_start)

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