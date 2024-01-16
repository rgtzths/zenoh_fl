#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Gabriele Baldoni'
__version__ = '0.1'
__email__ = 'gabriele@zettascale.tech'
__status__ = 'Development'

import json
import pathlib
import time
import logging
import numpy as np
import tensorflow as tf
from ZENOH.zcomm import ZComm, ALL_SRC, ANY_SRC, ANY_TAG
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG)

tf.keras.utils.set_random_seed(42)


def run(
    dataset_util, 
    optimizer,
    early_stop,
    learning_rate,
    batch_size, 
    global_epochs, 
    local_epochs, 
    alpha,
    patience,
    min_delta,
    n_workers,
    rank
):

    n_workers = comm.Get_size()-1
    stop = False
    dataset = dataset_util.name
    patience_buffer = [0]*patience

    output = f"{dataset}/zenoh/decentralized_async/{n_workers}_{global_epochs}_{local_epochs}_{alpha}"
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = pathlib.Path(dataset)
    
    model = dataset_util.create_model()
    model.compile(
        optimizer=optimizer(learning_rate=learning_rate), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    start = time.time()
    comm = ZComm(rank, n_workers)

    logging.info(f'[RANK: {rank}] Waiting nodes...')
    comm.wait(n_workers+1)
    logging.info(f'[RANK: {rank}] Nodes up!')

    '''
    Initial configuration, the parameter server receives the amount of 
    examples each worker has to perform a
    weighted average of their contributions.
    '''

    if rank == 0:
        results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "sync" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "global_times" : []}}

        node_weights = [0]*(n_workers)
        X_cv, y_cv = dataset_util.load_validation_data()

        val_dataset = tf.data.Dataset.from_tensor_slices(X_cv).batch(batch_size)

        #Get the amount of training examples of each worker and divides it by the total
        #of examples to create a weighted average of the model weights
        # for node in range(n_workers):
        #         status = MPI.Status()
        #         comm.Recv(buff, source=MPI.ANY_SOURCE, tag=1000, status=status)
        #         n_examples = pickle.loads(buff)
        data = comm.recv(source=ALL_SRC, tag=1000)
        for (source,src_tag), n_examples in data.items():
            # node_weights[status.Get_source()-1] = n_examples
            node_weights[source-1] = n_examples
        
        biggest_n_examples = max(node_weights)

        node_weights = [n_examples/biggest_n_examples for n_examples in node_weights]

        results["times"]["sync"].append(time.time() - start)
        # weights = bytearray(pickle.dumps(model.get_weights()))

    else:
        results = {"times" : {"train" : [], "comm_send" : [], "comm_recv" : [], "conv_send" : [], "conv_recv" : [], "epochs" : []}}

        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

        # compr_data = pickle.dumps(len(X_train))

        # comm.Send(compr_data, dest=0, tag=1000)
        comm.send(dest=0, data=len(X_train), tag=1000)
        # weights = buff


    '''
    Parameter server shares its values so every worker starts from the same point.
    '''
    # comm.Bcast(weights, root=0)
    model.set_weights(comm.bcast(data=model.get_weights(), root=0, tag=0))

    if rank == 0:
        results["times"]["sync"].append(time.time() - start)

    '''
    Training starts.
    Rank 0 is the responsible for aggregating the weights of the models
    The remaining perform training
    '''
    if rank == 0:
        local_weights = model.get_weights()
        exited_workers = 0
        epoch_start = time.time()

        for epoch in range(global_epochs*(n_workers)):
            if epoch % n_workers == 0:

                logging.info("Start of epoch %d" % (epoch//n_workers+1))
            
            #This needs to be changed to the correct formula
            
            # com_time = time.time()
            # comm.Recv(buff, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            # results["times"]["comm_recv"].append(time.time() - com_time)

            # load_time = time.time()
            # weights = pickle.loads(buff)
            # results["times"]["conv_recv"].append(time.time() - load_time)

            # source = status.Get_source()
            # tag = status.Get_tag()

            #Check how to combine here
            data = comm.recv(source=ANY_SRC, tag=ANY_TAG)
            for (source, src_tag), weights in data.items():
                weight_diffs = [ (weight - local_weights[idx])*alpha*node_weights[source-1]
                                for idx, weight in enumerate(weights)]
                
                local_weights = [local_weights[idx] + weight
                                for idx, weight in enumerate(weight_diffs)]
            
            # load_time = time.time()
            # weights = pickle.dumps(weight_diffs)
            # results["times"]["conv_send"].append(time.time() - load_time)
            
            # comm_time = time.time()
            comm.send(data=weight_diffs, dest=source, tag=src_tag)
            # comm.Send(weights, dest=source, tag=tag)
            # results["times"]["comm_send"].append(time.time() - comm_time)

            comm.send(stop, dest=source, tag=src_tag)

            if stop:
                exited_workers +=1
            if exited_workers == n_workers:
                break

            if epoch % n_workers == n_workers-1:
                results["times"]["epochs"].append(time.time() - epoch_start)
                model.set_weights(local_weights)
                predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
                val_f1 = f1_score(y_cv, predictions, average="macro")
                val_mcc = matthews_corrcoef(y_cv, predictions)
                val_acc = accuracy_score(y_cv, predictions)

                results["acc"].append(val_acc)
                results["f1"].append(val_f1)
                results["mcc"].append(val_mcc)
                results["times"]["global_times"].append(time.time() - start)
                patience_buffer = patience_buffer[1:]
                patience_buffer.append(val_mcc)

                logging.info("- val_f1: %6.3f - val_mcc %6.3f - val_acc %6.3f" %(val_f1, val_mcc, val_acc))

                if val_mcc > early_stop or abs(patience_buffer[0] - patience_buffer[-1]) < min_delta:
                    stop = True

                epoch_start = time.time()

    else:
        for global_epoch in range(global_epochs):
            epoch_start = time.time()

            train_time = time.time()
            model.fit(train_dataset, epochs=local_epochs, verbose=0)
            results["times"]["train"].append(time.time() - train_time)

            # load_time = time.time()
            # weights = pickle.dumps(model.get_weights())
            # results["times"]["conv_send"].append(time.time() - load_time)
            
            # comm_time = time.time()
            # comm.Send(weights, dest=0, tag=global_epoch)
            comm.send(data=model.get_weights(), dest=0, tag=global_epoch)
            # results["times"]["comm_send"].append(time.time() - comm_time)

            # com_time = time.time()
            # comm.Recv(buff, source=0, tag=global_epoch)
            # results["times"]["comm_recv"].append(time.time() - com_time)
            data = comm.recv(source=0, tag=global_epoch)
            for (s, t), weight_diffs in data.items():
                # results["times"]["conv_recv"].append(time.time() - load_time)

                weights = [weight - weight_diffs[idx]
                                for idx, weight in enumerate(model.get_weights())]
                
                model.set_weights(weights)    
            
            # results["times"]["epochs"].append(time.time() - epoch_start)

            # load_time = time.time()
            # weight_diffs = pickle.loads(buff)
            # results["times"]["conv_recv"].append(time.time() - load_time)

            # weights = [weight - weight_diffs[idx]
            #                  for idx, weight in enumerate(model.get_weights())]
            
            # model.set_weights(weights)    
            
            # results["times"]["epochs"].append(time.time() - epoch_start)
            stop = comm.recv(source=0, tag=global_epoch)

            if stop:
                break

    history = json.dumps(results)
    if rank==0:
        model.save(output/'trained_model.keras')
        with open(output/"server.json", "w") as f:
            f.write(history)

    else:
        with open(output/f"worker{rank}.json", "w") as f:
            f.write(history)
    comm.close()