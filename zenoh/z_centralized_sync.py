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
import zenoh
from zenoh import config, Value, Reliability, Sample, CongestionControl
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import pickle


ALL_SRC = -1
ANY_SRC = -2

class ZCommData(object):
    def __init__(self, src, dest, tag, data):
        self.src = src
        self.dest = dest
        self.tag = tag
        self.data = data

    def serialize(self):
        return pickle.dumps(self)

    def deserialize(data):
        return pickle.loads(data)
    
    def __str__(self):
        return f'SRC: {self.src} DST: {self.dest} TAG: {self.tag}'


class ZComm(object):
    def __init__(self, rank, workers, locator =  None):
        self.session = self.connect_zenoh(locator)
        self.rank = rank
        self.data = {}

        self.ke_live_queryable = f'mpi/{rank}/status'
        self.ke_data = f'mpi/{rank}/data'
        self.expected = workers+1

        self.queryable = self.session.declare_queryable(self.ke_data, self.data_cb)
        self.live_queryable = self.session.declare_queryable(self.ke_live_queryable, self.live_cb)

    def connect_zenoh(self, locator):
        conf = zenoh.Config()
        conf.insert_json5(zenoh.config.MODE_KEY, json.dumps("peer"))
        if locator is not None:
            conf.insert_json5(zenoh.config.CONNECT_KEY, json.dumps([locator]))
        zenoh.init_logger()
        session = zenoh.open(conf)
        return session

    def close(self):
        self.queryable.undeclare()
        self.live_queryable.undeclare()
        self.session.close()


    def update_data(self, zcomdata):
        src_data = self.data.get(zcomdata.src)
        if src_data is None:
            # No data received from this source
            # creating the inner dict
            data_dict = {zcomdata.tag: zcomdata.data}
            self.data.update({zcomdata.src:data_dict})
        else:
            src_data.update({zcomdata.tag: zcomdata.data})

    def data_cb(self, query):
        ke = f'{query.selector.key_expr}'
        data = ZCommData.deserialize(query.value.payload)
        # print(f'[RANK {self.rank}] Received on {ke} - Data: {data}')
        self.update_data(data)
        query.reply(Sample(ke, b''))

    def live_cb(self, query):
        data = f'{self.rank}-up'.encode("utf-8")
        query.reply(Sample(self.ke_live_queryable, data))


    def wait(self, expected):
        for i in range(0, expected):
            if i == self.rank:
                continue
            ke = f'mpi/{i}/status'
            data = None
            while data is None:
                replies = self.session.get(ke, zenoh.Queue())
                for reply in replies.receiver:
                    if int(reply.ok.payload.decode("utf-8").split("-")[0]) == i:
                        data = "up"
                time.sleep(0.05)

    def recv(self, source, tag):
        if source == ALL_SRC:
            return self.recv_from_all(tag)
        elif source == ANY_SRC:
            return self.recv_from_any(tag)

        expected = 1
        acks = 0

        while acks < expected:
            src_data = self.data.get(source)
            if src_data is None:
                time.sleep(0.005)
                continue
            tag_data = src_data.get(tag)
            if tag_data is None:
                time.sleep(0.005)
                continue
            del src_data[tag]
            acks = 1
            return {source: tag_data}

    def send(self, dest, data, tag):
        msg = ZCommData(self.rank, dest, tag, data)
        expected = 1
        acks = 0
        ke = f"mpi/{dest}/data"
        # print(f'[RANK: {self.rank}] Sending on {ke} - Data: {msg}')
        while acks < expected:
            replies = self.session.get(ke, zenoh.Queue(), value = msg.serialize())
            for reply in replies:
                try:
                    if reply.ok is not None:
                        acks = 1
                except:
                    continue
            
            time.sleep(0.005)

    def bcast(self, root, data, tag):
        if self.rank == root:
            for i in range(0, self.expected):
                if i == self.rank:
                    # do not self send
                    continue
                self.send(i, data, tag)
            return data
        else:
            recv_data = self.recv(root, tag)
            return recv_data[root]
        
    def recv_from_all(self, tag):
        data = {}
        for i in range(0, self.expected):
            if i == self.rank:
                # do not recv from self
                continue
            data.update(self.recv(i, tag))
        return data

    def recv_from_any(self, tag):
        return []



def create_MLP():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(60,)))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))

    return model

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

logging.debug(f'[RANK: {rank}] Waiting nodes...')
comm.wait(n_workers+1)
logging.debug(f'[RANK: {rank}] Nodes up!')

# if rank == 0:
#     data = comm.recv(1, 100)
#     print(f'Data is: {data}')
#     from_all_data = comm.recv(ANY_SRC, 101)
#     print(f'Data from ANY is: {from_all_data}')


# else:
#     comm.send(0, "Hello, from Worker", 100)
#     print(f'[RANK: {rank}] Data sent')
#     comm.send(0, "new Hello, from Worker", 101)
#     print(f'[RANK: {rank}] Data sent')



# bcast_data = comm.bcast(0, "Broadcasted data", 0)
# print(f'[RANK: {rank}] Bcast received data {bcast_data}')

# time.sleep(5)
# comm.close()
# exit(0)

if rank == 0:
    node_weights = [0]*n_workers
    X_cv = np.loadtxt(dataset/"x_cv.csv", delimiter=",", dtype=int)
    y_cv = np.loadtxt(dataset/"y_cv.csv", delimiter=",", dtype=int)

    val_dataset = tf.data.Dataset.from_tensor_slices(X_cv).batch(batch_size)

    #Get the amount of training examples of each worker and divides it by the total
    #of examples to create a weighted average of the model weights
    # for node in range(n_workers):
        
        # n_examples = comm.recv(source=MPI.ANY_SOURCE, tag=1000, status=status)
        # node_weights[status.Get_source()-1] = n_examples
    data = comm.recv(source=ALL_SRC, tag=1000)
    for k, v in data.items():
        node_weights[k-1] = v
    # print(f'[RANK: {rank}] node_weights: {node_weights}!')
    
    
    total_size = sum(node_weights)

    node_weights = [weight/total_size for weight in node_weights]
    results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "loads" : []}}
    results["times"]["loads"].append(time.time() - start)
    # print(f'[RANK: {rank}] Results: {results}!')

else:
    X_train = np.loadtxt(dataset/("x_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = np.loadtxt(dataset/("y_train_subset_%d.csv" % rank), delimiter=",", dtype=int)
    y_train = tf.keras.utils.to_categorical(y_train)

    train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

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
        for source, grads in grads_recv.items(): 
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
        batch = (batch + 1) % len(train_dataset)

    model.set_weights(comm.bcast(data=weights, root=0, tag=0))

    if rank == 0 and epoch % 1500 == 0:
        logging.debug("\n End of epoch %d" % epoch)
        predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
        train_f1 = f1_score(y_cv, predictions, average="macro")
        train_mcc = matthews_corrcoef(y_cv, predictions)
        train_acc = accuracy_score(y_cv, predictions)

        results["acc"].append(train_acc)
        results["f1"].append(train_f1)
        results["mcc"].append(train_mcc)
        results["times"]["epochs"].append(time.time() - start)
        logging.debug("- val_f1: %f - val_mcc %f - val_acc %f" %(train_f1, train_mcc, train_acc))

    
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()

if rank==0:
    history = json.dumps(results)
    logging.debug(f'Saving results in {output}')
    f = open( output/"train_history.json", "w")
    f.write(history)
    f.close()

    model.save(output/'trained_model.h5')

comm.close()
