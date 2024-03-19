import json
import pathlib
import time
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import pickle
import sys

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
    output
):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_workers = comm.Get_size()-1
    status = MPI.Status()
    stop = False
    dataset = dataset_util.name
    patience_buffer = [-1]*patience
    model_weights = None


    if rank == 0:
        print("Running decentralized async")
        print(f"Dataset: {dataset}")
        print(f"Learning rate: {learning_rate}")
        print(f"Global epochs: {global_epochs}")
        print(f"Local epochs: {local_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Alpha: {alpha}")

    output = f"{output}/{dataset}/mpi/decentralized_async/{n_workers}_{global_epochs}_{local_epochs}_{alpha}_{batch_size}"
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = pathlib.Path(dataset)
    
    model = dataset_util.create_model()
    model.compile(
        optimizer=optimizer(learning_rate=learning_rate), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    sent_size = 0
    received_size = 0

    start = time.time()

    '''
    Initial configuration, the parameter server receives the amount of 
    examples each worker has to perform a
    weighted average of their contributions.
    '''
    if rank == 0:
        results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "global_times" : []}}

        node_weights = [0]*(n_workers)
        
        X_cv, y_cv = dataset_util.load_validation_data()

        val_dataset = tf.data.Dataset.from_tensor_slices(X_cv).batch(batch_size)

        #Get the amount of training examples of each worker and divides it by the total
        #of examples to create a weighted average of the model weights
        for _ in range(n_workers):
            n_examples = comm.recv(source=MPI.ANY_SOURCE, tag=1000, status=status)
            node_weights[status.Get_source()-1] = n_examples
            received_size += sys.getsizeof(pickle.dumps(n_examples))

        
        biggest_n_examples = max(node_weights)

        node_weights = [n_examples/biggest_n_examples for n_examples in node_weights]
        model_weights = model.get_weights()

    else:
        results = {"times" : {"train" : [], "epochs" : []}}

        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

        comm.send(len(train_dataset), dest=0, tag=1000)

    '''
    Parameter server shares its values so every worker starts from the same point.
    '''
    model_weights = comm.bcast(model_weights, root=0)

    if rank == 0:
        model_size = sys.getsizeof(pickle.dumps(model_weights))
        sent_size += model_size*n_workers 
    else:
        model.set_weights(model_weights)

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

                print("\nStart of epoch %d, elapsed time %5.1fs" % (epoch//n_workers+1, time.time() - start))
            
            #This needs to be changed to the correct formula
            
            weights = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            received_size+=sys.getsizeof(pickle.dumps(weights))

            source = status.Get_source()
            tag = status.Get_tag()

            #Check how to combine here
            weight_diffs = [ (weight - local_weights[idx])*alpha*node_weights[source-1]
                            for idx, weight in enumerate(weights)]
            
            local_weights = [local_weights[idx] + weight
                            for idx, weight in enumerate(weight_diffs)]
            
            comm.send(weight_diffs, dest=source, tag=tag)
            sent_size += sys.getsizeof(pickle.dumps(weight_diffs)) 

            comm.send(stop, dest=source, tag=tag)
            sent_size += sys.getsizeof(pickle.dumps(stop)) 

            if stop:
                exited_workers +=1
            if exited_workers == n_workers:
                break

            if epoch % n_workers == n_workers-1 and not stop:
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

                print("- val_f1: %6.3f - val_mcc %6.3f - val_acc %6.3f - sent_messages  %6.3f - received_messages  %6.3f "  %(val_f1, val_mcc, val_acc, sent_size*0.000001, received_size*0.000001))

                p_stop = True
                for value in patience_buffer[1:]:
                    if abs(patience_buffer[0] - value) > min_delta:
                        p_stop = False 

                if val_mcc > early_stop or p_stop:
                    stop = True

                epoch_start = time.time()
    else:
        for global_epoch in range(global_epochs):
            epoch_start = time.time()

            train_time = time.time()
            model.fit(train_dataset, epochs=local_epochs, verbose=0)
            results["times"]["train"].append(time.time() - train_time)

            comm.send(model.get_weights(), dest=0, tag=global_epoch)

            weight_diffs = comm.recv(source=0, tag=global_epoch)

            stop = comm.recv(source=0, tag=global_epoch)

            weights = [weight - weight_diffs[idx]
                            for idx, weight in enumerate(model.get_weights())]
            
            model.set_weights(weights)    
            
            results["times"]["epochs"].append(time.time() - epoch_start)

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