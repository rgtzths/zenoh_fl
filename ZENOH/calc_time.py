#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import sys
sys.path.append('.')

import pickle
import time
import sys
import time
import logging
import tensorflow as tf
from zcomm import ZCommPy
import argparse
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG)
from config import DATASETS, OPTIMIZERS
import asyncio


tf.keras.utils.set_random_seed(42)
parser = argparse.ArgumentParser()
parser.add_argument("-nw", help="Number of workers (for zenoh)", default=2, type=int)
parser.add_argument("-wid", help="Worker id (for zenoh)", default=0, type=int)
args = parser.parse_args()

async def run():
    comm = await ZCommPy.new(args.wid, args.nw, "tcp/127.0.0.1:7447")
    comm.start()

    rank = args.wid
    n_workers = args.nw

    logging.info(f'[RANK: {rank}] Waiting nodes...')
    await comm.wait()
    logging.info(f'[RANK: {rank}] Nodes up!')

    for dataset in DATASETS:
        print(f"Sending model for dataset: {dataset}")
        time.sleep(5)
        '''
        model loading
        '''
        model = DATASETS[dataset](42).create_model()
        model.compile(
            optimizer=OPTIMIZERS["Adam"](learning_rate=0.001), 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        if rank == 0:
            '''
            Model size
            '''
            print(f"Size of the model: {sys.getsizeof(pickle.dumps(model.get_weights()))*0.000001:.2f}")
            
            '''
            Measuring comm time
            '''
            start_time = time.time()
            await comm.send(dest=1, tag=-10, data=pickle.dumps(model.get_weights()))
            end_time = time.time() - start_time
            print("Time to send the model weights: ", end_time/60)
        else:
            data = await comm.recv(src=0, tag=-10)
            for source, message in data.items():
                model = pickle.loads(message.data)

asyncio.run(run())