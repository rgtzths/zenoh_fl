from transformers import BertTokenizer, TFBertModel
import pickle
from mpi4py import MPI
import time
import sys

'''
MPI stuff
'''
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_workers = comm.Get_size()-1
status = MPI.Status()

'''
model loading
'''
model = TFBertModel.from_pretrained("bert-base-uncased")

if rank == 0:
    '''
    Model size
    '''
    print(f"Size of the model: {sys.getsizeof(pickle.dumps(model.get_weights()))*0.000001:.2f}")
    
    '''
    Measuring comm time
    '''
    start_time = time.time()
    comm.send(model.get_weights(), dest=1, tag=1)
    end_time = time.time() - start_time
    print("Time to send the model weights: ", end_time/60)
else:
    model = comm.recv(source=0, tag=1)

