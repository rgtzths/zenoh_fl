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
    epochs,
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
        print("Running centralized sync")
        print(f"Dataset: {dataset}")
        print(f"Learning rate: {learning_rate}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")

    output = f"{output}/{dataset}/mpi/centralized_sync/{n_workers}_{epochs}_{batch_size}"
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = pathlib.Path(dataset)

    model = dataset_util.create_model()
    optimizer = optimizer(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    sent_size = 0
    received_size = 0

    start = time.time()
    if rank == 0:
        results = {"acc" : [], "mcc" : [], "f1" : [],  "messages_size" : {"sent" : [], "received" : []}, "times" : {"epochs" : [], "global_times" : []}}
        node_weights = [0]*n_workers

        X_cv, y_cv = dataset_util.load_validation_data()
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_cv, y_cv)).batch(batch_size)

        #Get the amount of training examples of each worker and divides it by the total
        #of examples to create a weighted average of the model weights
        for node in range(n_workers):
            n_examples = comm.recv(source=MPI.ANY_SOURCE, tag=1000, status=status)
            received_size += sys.getsizeof(pickle.dumps(n_examples))

            node_weights[status.Get_source()-1] = n_examples
        
        sum_n_batches = sum(node_weights)
        total_n_batches = max(node_weights)
        total_batches = epochs * total_n_batches

        node_weights = [weight/sum_n_batches for weight in node_weights]
        model_weights = model.get_weights()

    else:
        results = {"times" : {"train" : [], "epochs" : []}}

        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)

        train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))

        comm.send(len(train_dataset), dest=0, tag=1000)

        total_n_batches = len(train_dataset)
        total_batches = epochs * total_n_batches


    model_weights = comm.bcast(model_weights, root=0)

    if rank == 0:
        sent_size += sys.getsizeof(pickle.dumps(model_weights))*n_workers 
    else:
        model.set_weights(model_weights)

    epoch_start = time.time()
    for batch in range(total_batches):
        if rank == 0:
            avg_grads = []
            for _ in range(n_workers):
                grads = comm.recv(source=MPI.ANY_SOURCE, tag=batch, status=status)
                
                received_size+=sys.getsizeof(pickle.dumps(grads))

                source = status.Get_source()

                if not avg_grads:
                    avg_grads = [grad*node_weights[source-1] for grad in grads]
                else:
                    for idx, weight in enumerate(grads):
                        avg_grads[idx] += weight*node_weights[source-1]
            
            optimizer.apply_gradients(zip(avg_grads, model.trainable_weights))

            model_weights = model.get_weights()    
            sent_size += sys.getsizeof(pickle.dumps(model_weights))*n_workers 
        else:
            train_time = time.time()

            x_batch_train, y_batch_train = train_dataset[batch % len(train_dataset)]

            with tf.GradientTape() as tape:

                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            results["times"]["train"].append(time.time() - train_time)

            comm.send(grads, dest=0, tag=batch)

        model_weights = comm.bcast(model_weights, root=0)
        
        stop = comm.bcast(stop, root=0)
        if rank == 0:
            sent_size += sys.getsizeof(pickle.dumps(stop))*8

        if rank != 0:
            model.set_weights(model_weights)

        if stop:
            break

        if (batch+1) % total_n_batches == 0:

            if rank == 0:
                results["times"]["epochs"].append(time.time() - epoch_start)

                print(f"\n End of batch {batch+1} -> epoch {(batch+1) // total_n_batches}, elapsed time {time.time() - start:.1f}s")
                predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
                val_f1 = f1_score(y_cv, predictions, average="macro")
                val_mcc = matthews_corrcoef(y_cv, predictions)
                val_acc = accuracy_score(y_cv, predictions)

                results["acc"].append(val_acc)
                results["f1"].append(val_f1)
                results["mcc"].append(val_mcc)
                results["messages_size"]["sent"].append(sent_size)
                results["messages_size"]["received"].append(received_size)
                results["times"]["global_times"].append(time.time() - start)
                print("- val_f1: %6.3f - val_mcc %6.3f - val_acc %6.3f - sent_messages  %6.3f - received_messages  %6.3f "  %(val_f1, val_mcc, val_acc, sent_size*0.000001, received_size*0.000001))
                patience_buffer = patience_buffer[1:]
                patience_buffer.append(val_mcc)

                p_stop = True
                for value in patience_buffer[1:]:
                    if abs(patience_buffer[0] - value) > min_delta:
                        p_stop = False 

                if val_mcc > early_stop or p_stop:
                    stop = True

            else:
                results["times"]["epochs"].append(time.time() - epoch_start)

            epoch_start = time.time()


    history = json.dumps(results)
    if rank==0:
        model.save(output/'trained_model.keras')
        with open(output/"server.json", "w") as f:
            f.write(history)

    else:
        with open(output/f"worker{rank}.json", "w") as f:
            f.write(history)
