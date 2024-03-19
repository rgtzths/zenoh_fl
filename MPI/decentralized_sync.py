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
    patience,
    min_delta,
    output
):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_workers = comm.Get_size()-1
    status = MPI.Status()
    stop = False
    model_weights = None

    dataset = dataset_util.name
    patience_buffer = [-1]*patience

    if rank == 0:
        print("Running decentralized sync")
        print(f"Dataset: {dataset}")
        print(f"Learning rate: {learning_rate}")
        print(f"Global epochs: {global_epochs}")
        print(f"Local epochs: {local_epochs}")
        print(f"Batch size: {batch_size}")

    output = f"{output}/{dataset}/mpi/decentralized_sync/{n_workers}_{global_epochs}_{local_epochs}_{batch_size}"
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

    if rank == 0:
        results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "global_times" : []}}
        node_weights = [0]*(n_workers)
        X_cv, y_cv = dataset_util.load_validation_data()

        val_dataset = tf.data.Dataset.from_tensor_slices(X_cv).batch(batch_size)

        #Get the amount of training examples of each worker and divides it by the total
        #of examples to create a weighted average of the model weights
        for node in range(n_workers):
            n_examples = comm.recv(source=MPI.ANY_SOURCE, tag=1000, status=status)
            node_weights[status.Get_source()-1] = n_examples
            received_size += sys.getsizeof(pickle.dumps(n_examples))

        total_size = sum(node_weights)

        node_weights = [weight/total_size for weight in node_weights]

        model_weights = model.get_weights()

    else:
        results = {"times" : {"train" : [], "epochs" : []}}

        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)        

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

        comm.send(len(train_dataset), dest=0, tag=1000)

    model_weights = comm.bcast(model_weights, root=0)

    if rank == 0:
        sent_size += sys.getsizeof(pickle.dumps(model_weights))*n_workers 
    else:
        model.set_weights(model_weights)

    for global_epoch in range(global_epochs):
        avg_weights = []
        epoch_start = time.time()

        if rank == 0:
            print("\nStart of epoch %d, elapsed time %5.1fs" % (global_epoch+1, time.time() - start))

            for _ in range(n_workers):
                weights = comm.recv(source=MPI.ANY_SOURCE, tag=global_epoch, status=status)
                received_size+=sys.getsizeof(pickle.dumps(weights))

                source = status.Get_source()

                if not avg_weights:
                    avg_weights = [ weight * node_weights[source-1] for weight in weights]
                else:
                    avg_weights = [ avg_weights[i] + weights[i] * node_weights[source-1] for i in range(len(weights))]
            sent_size += sys.getsizeof(pickle.dumps(avg_weights))*n_workers        
        else:
            train_time = time.time()
            model.fit(train_dataset, epochs=local_epochs, verbose=0)
            results["times"]["train"].append(time.time() - train_time)

            comm.send(model.get_weights(), dest=0, tag=global_epoch)

        avg_weights = comm.bcast(avg_weights, root=0)

        stop = comm.bcast(stop, root=0)
        
        if rank != 0:
            model.set_weights(avg_weights)
            results["times"]["epochs"].append(time.time() - epoch_start)
            if stop:
                break
        else:
            sent_size += sys.getsizeof(pickle.dumps(stop))*8

            results["times"]["epochs"].append(time.time() - epoch_start)

            model.set_weights(avg_weights)

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
            if stop:
                break
            p_stop = True
            for value in patience_buffer[1:]:
                if abs(patience_buffer[0] - value) > min_delta:
                    p_stop = False 

            if val_mcc > early_stop or p_stop:
                stop = True

    history = json.dumps(results)
    if rank==0:
        model.save(output/'trained_model.keras')
        with open(output/"server.json", "w") as f:
            f.write(history)

    else:
        with open(output/f"worker{rank}.json", "w") as f:
            f.write(history)