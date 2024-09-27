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
    patience_buffer = [0]*patience

    if rank == 0:
        print("Running centralized sync")
        print(f"Dataset: {dataset}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")

    output = f"{output}/{dataset}/{dataset_util.seed}/mpi/centralized_sync/{n_workers}_{epochs}_{batch_size}"
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = pathlib.Path(dataset)

    model = dataset_util.create_model()
    optimizer = optimizer(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    if rank == 0:
        results = {"acc" : [], "mcc" : [], "f1" : []}
        node_weights = [1/n_workers]*n_workers

        X_cv, y_cv = dataset_util.load_validation_data()
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_cv, y_cv)).batch(8192)

        total_n_batches = 0
        for node in range(n_workers):
            n_batches = comm.recv(source=MPI.ANY_SOURCE, tag=1000, status=status)
            total_n_batches += n_batches
        

    else:
        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)

        train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))

        comm.send(len(train_dataset), dest=0, tag=1000)

        total_n_batches = len(train_dataset)


    weights = comm.bcast(model.get_weights(), root=0)

    if rank != 0:
        model.set_weights(weights)

    for batch in range(total_n_batches*epochs):
        weights = []
        if rank == 0:
            avg_grads = []
            for _ in range(n_workers):

                grads = comm.recv(source=MPI.ANY_SOURCE, tag=batch, status=status)

                source = status.Get_source()

                if not avg_grads:
                    avg_grads = [grad*node_weights[source-1] for grad in grads]
                else:
                    for idx, weight in enumerate(grads):
                        avg_grads[idx] += weight*node_weights[source-1]
            
            optimizer.apply_gradients(zip(avg_grads, model.trainable_weights))
            
        else:
            x_batch_train, y_batch_train = train_dataset[batch % len(train_dataset)]

            with tf.GradientTape() as tape:

                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)

            comm.send(grads, dest=0, tag=batch)
        weights = comm.bcast(model.get_weights(), root=0)

        if rank != 0:
            model.set_weights(weights)
        
        stop = comm.bcast(stop, root=0)

        if stop:
            break

        if (batch+1) % total_n_batches == 0:

            if rank == 0:
                print(f"\n End of batch {batch+1} -> epoch {(batch+1) // total_n_batches}")
                predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
                val_f1 = f1_score(y_cv, predictions, average="weighted")
                val_mcc = matthews_corrcoef(y_cv, predictions)
                val_acc = accuracy_score(y_cv, predictions)

                results["acc"].append(val_acc)
                results["f1"].append(val_f1)
                results["mcc"].append(val_mcc)
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

    if rank==0:
        history = json.dumps(results)
        model.set_weights(best_weights)
        model.save(output/'centralized_sync.keras')
        with open(output/"server.json", "w") as f:
            f.write(history)