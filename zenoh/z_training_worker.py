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

def connect_zenoh():
    conf = zenoh.Config()
    conf.insert_json5(zenoh.config.MODE_KEY, json.dumps("peer"))
    #conf.insert_json5(zenoh.config.CONNECT_KEY, json.dumps([locator]))
    zenoh.init_logger()
    session = zenoh.open(conf)
    return session

def create_MLP(learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(60,)))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class TrainingWorker(object):
    def __init__(self, global_epochs, local_epochs, batch_size, learning_rate, dataset, output, node_id):
        tf.keras.utils.set_random_seed(7)
        self.session = connect_zenoh()
        self.output = pathlib.Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.dataset = pathlib.Path(dataset)
        self.model = create_MLP(learning_rate)
        
        self.global_epochs = global_epochs
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.id = node_id


        self.node_weights = []
        self.X_train = None
        self.Y_train = None
        self.val_dataset = None
        self.total_size = None 
        
        self.latest_value = None

        self.sub = self.session.declare_subscriber("manager/weights", self.subscriber_cb, reliability=Reliability.RELIABLE())
        self.pub  = self.session.declare_publisher(f"{self.id}/weights")

        self.start = None
        logging.debug("[TrainingWorker] initialized")
        
    
    def subscriber_cb(self, sample):
        logging.debug(f">> [Subscriber] Received {sample.kind} ('{sample.key_expr}': {len(sample.payload)} bytes)")
        self.latest_value = pickle.loads(sample.payload)

    def recv_from_manager(self):
        while self.latest_value == None:
            time.sleep(0.0001)
        val = self.latest_value
        self.latest_value = None
        return val


    def run(self):
        self.start = time.time()
        logging.debug(f'Start time {self.start}')
        self.X_train = np.loadtxt(self.dataset/("x_train_subset_%d.csv" % self.id), delimiter=",", dtype=int)
        self.Y_train = np.loadtxt(self.dataset/("y_train_subset_%d.csv" % self.id), delimiter=",", dtype=int)
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train)

        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.Y_train)).batch(self.batch_size)

        self.pub.put(pickle.dumps({'node':self.id, 'len':len(self.Y_train)}))
        new_weights = self.recv_from_manager()
        logging.debug(f"Initial weights: {new_weights[:1]}")

        self.model.set_weights(new_weights)

        for global_epoch in range(self.global_epochs):
            self.model.fit(train_dataset, epochs=local_epochs, verbose=0)
            self.pub.put(pickle.dumps({'node':self.id, 'tag':global_epoch, 'weights': self.model.get_weights()}))

            new_weights = self.recv_from_manager()
            logging.debug(f"New weights: {new_weights[:1]}")
            self.model.set_weights(new_weights)

            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            gc.collect()


if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='Train and test the model')
    parser.add_argument('--g_epochs', type=int, help='Global epochs number', default=10)
    parser.add_argument('--l_epochs', type=int, help='local epochs number', default=5)
    parser.add_argument('-b', type=int, help='Batch size', default=64)
    parser.add_argument('-l', type=float, help='Learning rate', default=0.00001)
    parser.add_argument('-d', type=str, help='Dataset', default="one_hot/")
    parser.add_argument('-o', type=str, help='Output folder', default="results")
    parser.add_argument('-i', type=int, help='Worker id')

    args = parser.parse_args()

    global_epochs = args.g_epochs
    local_epochs = args.l_epochs
    batch_size = args.b
    learning_rate = args.l
    dataset = args.d
    output = args.o

    worker = TrainingWorker(global_epochs, local_epochs, batch_size, learning_rate, dataset, output, args.i)
    worker.run()



