from transformers import BertTokenizer, TFBertModel
import pickle
from mpi4py import MPI
import time
import sys

#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import time
import logging
import numpy as np
import tensorflow as tf
from zcomm import ZComm, ALL_SRC, ANY_SRC, ANY_TAG
import argparse
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG)

tf.keras.utils.set_random_seed(42)
'''
model loading
'''
model = TFBertModel.from_pretrained("bert-base-uncased")


parser = argparse.ArgumentParser()
parser.add_argument("-nw", help="Number of workers (for zenoh)", default=4, type=int)
parser.add_argument("-wid", help="Worker id (for zenoh)", default=0, type=int)
args = parser.parse_args()

comm = ZComm(args.wid, args.nw)

rank = args.wid
n_workers = args.nw

logging.info(f'[RANK: {rank}] Waiting nodes...')
comm.wait(n_workers+1)
logging.info(f'[RANK: {rank}] Nodes up!')

if rank == 0:
    '''
    Model size
    '''
    print(f"Size of the model: {sys.getsizeof(pickle.dumps(model.get_weights()))*0.000001:.2f}")
    
    '''
    Measuring comm time
    '''
    start_time = time.time()
    comm.send(data=model.get_weights(), dest=1, tag=1)
    end_time = time.time() - start_time
    print("Time to send the model weights: ", end_time/60)
else:
    model = comm.recv(source=0, tag=1)

comm.close()