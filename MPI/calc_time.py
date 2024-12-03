import sys
sys.path.append('.')

# Add the current directory to the PATH
import pickle
from mpi4py import MPI
import time
import sys
import config
import tensorflow as tf
import time
tf.keras.utils.set_random_seed(42)


'''
MPI stuff
'''
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_workers = comm.Get_size()-1
status = MPI.Status()

for dataset in config.DATASETS:
    print(f"Sending model for dataset: {dataset}")
    time.sleep(5)
    '''
    model loading
    '''
    model = config.DATASETS[dataset](42).create_model()
    model.compile(
        optimizer=config.OPTIMIZERS["Adam"](learning_rate=0.001), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


    if rank == 0:
        '''
        Model size
        '''
        print(f"Size of the model: {sys.getsizeof(pickle.dumps(model.get_weights()))}")
        
        '''
        Measuring comm time
        '''
        start_time = time.time()
        comm.send(model.get_weights(), dest=1, tag=1)
        end_time = time.time() - start_time
        print("Time to send the model weights: ", end_time/60)
    else:
        model = comm.recv(source=0, tag=1)

    break