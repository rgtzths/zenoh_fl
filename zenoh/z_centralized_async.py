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
parser.add_argument('-e', type=int, help='Epochs number', default=10)
parser.add_argument('-l', type=float, help='Learning rate', default=0.00001)
parser.add_argument('-d', type=str, help='Dataset', default="one_hot/")
parser.add_argument('-o', type=str, help='Output folder', default="results")
parser.add_argument('-b', type=int, help='Batch size', default=1024)
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
    results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "sync" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "global_times" : []}}
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
    for (source,src_tag), v in data.items():
        #logging.debug(f'Data from {source, src_tag}')
        node_weights[source-1] = v
    
    total_n_batches = sum(node_weights)
    total_batches = epochs * total_n_batches

    node_weights = [weight/total_n_batches for weight in node_weights]
    results["times"]["sync"].append(time.time() - start)

else:
    results = {"times" : {"train" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "epochs" : []}}

    X_train = np.loadtxt(dataset/("x_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = np.loadtxt(dataset/("y_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = tf.keras.utils.to_categorical(y_train)

    train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    #comm.send(len(X_train), dest=0, tag=1000)
    comm.send(dest=0, data=len(train_dataset), tag=1000)
    #logging.debug(f'[RANK: {rank}] sent to {(0, 1000)}')

    total_batches = epochs * len(train_dataset)

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# model.set_weights(comm.bcast(model.get_weights(), root=0))
model.set_weights(comm.bcast(data=model.get_weights(), root=0, tag=0))


if rank == 0:
    results["times"]["sync"].append(time.time() - start)

epoch_start = time.time()
if rank == 0:
    #logging.debug(f'[RANK: {rank}] Expected loops {total_batches}')
    latest_tag = 0
    for batch in range(total_batches):
        #logging.debug(f'[RANK: {rank}] Batch {batch}')
        data = comm.recv(source=ANY_SRC, tag=ANY_TAG)
        for (source, src_tag), grads in data.items():

        # grads = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        # source = status.Get_source()
            #logging.debug(f'Data from {source, src_tag}')

            if latest_tag < src_tag+1:
                latest_tag = src_tag+1

            behind_penalty = (src_tag+1 / latest_tag) #The more behind it is the less impact it will have, verry small penalization

            grads = [grad*node_weights[source-1]*behind_penalty for grad in grads] 

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # comm.send(model.get_weights(), dest=source, tag=status.Get_tag())
            # TODO: return tag in recv
            comm.send(data=model.get_weights(), dest=source, tag=src_tag)
            #logging.debug(f'Sent to {source, src_tag}')

            if (batch+1) % total_n_batches == 0:
                results["times"]["epochs"].append(time.time() - epoch_start)

                logging.info(f"End of batch {(batch+1)//n_workers} -> epoch {(batch+1)//total_n_batches}")

                predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
                train_f1 = f1_score(y_cv, predictions, average="macro")
                train_mcc = matthews_corrcoef(y_cv, predictions)
                train_acc = accuracy_score(y_cv, predictions)

                results["acc"].append(train_acc)
                results["f1"].append(train_f1)
                results["mcc"].append(train_mcc)
                results["times"]["global_times"].append(time.time() - start)
                logging.info("- val_f1: %6.3f - val_mcc %6.3f - val_acc %6.3f" %(train_f1, train_mcc, train_acc))
                epoch_start = time.time()
                    
else:
    #logging.debug(f'[RANK: {rank}] Expected loops {total_batches}')
    for batch in range(total_batches):
        #logging.debug(f'[RANK: {rank}] epoch {batch}')
        x_batch_train, y_batch_train = train_dataset[batch % len(train_dataset)]

        with tf.GradientTape() as tape:

            logits = model(x_batch_train, training=True) 
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)

        #comm.send(grads, dest=0, tag=epoch)
        comm.send(data=grads, dest=0, tag=batch)
        #logging.debug(f'[RANK: {rank}] sent to {(0, batch)}')

        # model.set_weights(comm.recv(source=0, tag=epoch))
        data = comm.recv(source=0, tag=batch)
        for (s, t), v in data.items():
            #logging.debug(f'[RANK: {rank}] recv from {(s, t)}')

            model.set_weights(v)

if rank==0:
    history = json.dumps(results)
    logging.info(f'Saving results in {output}')
    f = open( output/"train_history.json", "w")
    f.write(history)
    f.close()

    model.save(output/'trained_model.h5')

comm.close()