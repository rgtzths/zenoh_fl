import json
import pathlib
import time
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


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

    tf.keras.utils.set_random_seed(dataset_util.seed)
    
    best_weights = None
    best_mcc = -1
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_workers = comm.Get_size()-1
    status = MPI.Status()
    stop = False
    dataset = dataset_util.name
    patience_buffer = [-1]*patience

    if rank == 0:
        print("Running centralized async")
        print(f"Dataset: {dataset}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")

    output = f"{output}/{dataset}/{dataset_util.seed}/mpi/centralized_async/{n_workers}_{epochs}_{batch_size}"
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = pathlib.Path(dataset)

    model = dataset_util.create_model()
    optimizer = optimizer(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    if rank == 0:
        results = {"acc" : [], "mcc" : [], "f1" : [], "messages_size" : {"sent" : [], "received" : []}, "times" : {"epochs" : [], "global_times" : []}}
        node_weights = [1/n_workers]*n_workers

        X_cv, y_cv = dataset_util.load_validation_data()

        val_dataset = tf.data.Dataset.from_tensor_slices((X_cv, y_cv)).batch(8192)

        total_n_batches = 0
        for node in range(n_workers):
            n_batches = comm.recv( source=MPI.ANY_SOURCE, tag=1000, status=status)
    
            total_n_batches += n_batches
        
    else:

        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)

        train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))

        comm.send(len(train_dataset), dest=0, tag=1000)

        total_n_batches = len(train_dataset)
    
    weights = comm.bcast(model.get_weights(), root=0)

    if rank != 0:
        model.set_weights(weights)

    start = time.time()
    if rank == 0:
        exited_workers = 0
        epoch_start = time.time()
        for batch in range(total_n_batches*epochs):

            grads = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

            source = status.Get_source()
            tag = status.Get_tag()

            grads = [grad*node_weights[source-1] for grad in grads] 

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            comm.send(model.get_weights(), dest=source, tag=tag)

            comm.send(stop, dest=source, tag=tag)

            if stop:
                exited_workers +=1
            if exited_workers == n_workers:
                break

            if (batch+1) % total_n_batches == 0 and not stop:


                print(f"\n End of batch {(batch+1)//n_workers} -> epoch {(batch+1)//total_n_batches}")

                predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
                val_f1 = f1_score(y_cv, predictions, average="weighted")
                val_mcc = matthews_corrcoef(y_cv, predictions)
                val_acc = accuracy_score(y_cv, predictions)

                results["acc"].append(val_acc)
                results["f1"].append(val_f1)
                results["mcc"].append(val_mcc)           
                results["times"]["global_times"].append(time.time() - start)
                results["times"]["epochs"].append(time.time() - epoch_start)

                print("- val_f1: %6.3f - val_mcc %6.3f - val_acc %6.3f" %(val_f1, val_mcc, val_acc))
                patience_buffer = patience_buffer[1:]
                patience_buffer.append(val_mcc)

                p_stop = True
                for value in patience_buffer[1:]:
                    if abs(patience_buffer[0] - value) > min_delta:
                        p_stop = False 

                if val_mcc >= early_stop or p_stop:
                    stop = True

                if val_mcc > best_mcc:
                    best_weights = model.get_weights()

                epoch_start = time.time()

    else:
        for batch in range(total_n_batches*epochs):

            x_batch_train, y_batch_train = train_dataset[batch % len(train_dataset)]

            with tf.GradientTape() as tape:

                logits = model(x_batch_train, training=True) 
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)

            comm.send(grads, dest=0, tag=batch)

            weights = comm.recv(source=0, tag=batch)

            stop = comm.recv(source=0, tag=batch)
            model.set_weights(weights)

            if stop:
                break

    if rank==0:
        history = json.dumps(results)
        model.set_weights(best_weights)
        model.save(output/'centralized_async.keras')
        with open(output/"server.json", "w") as f:
            f.write(history)

    
