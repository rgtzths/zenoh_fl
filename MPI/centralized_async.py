import json
import pathlib
import time
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

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
        print("Running centralized async")
        print(f"Dataset: {dataset}")
        print(f"Learning rate: {learning_rate}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")

    output = f"{output}/{dataset}/mpi/centralized_async/{n_workers}_{epochs}"
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = pathlib.Path(dataset)

    model = dataset_util.create_model()
    optimizer = optimizer(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    start = time.time()
    if rank == 0:
        results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "global_times" : []}}
        node_weights = [0]*n_workers

        X_cv, y_cv = dataset_util.load_validation_data()

        val_dataset = tf.data.Dataset.from_tensor_slices((X_cv, y_cv)).batch(batch_size)

        #Get the amount of training examples of each worker and divides it by the total
        #of examples to create a weighted average of the model weights
        for node in range(n_workers):
            n_examples = comm.recv(source=MPI.ANY_SOURCE, tag=1000, status=status)
            node_weights[status.Get_source()-1] = n_examples
        
        total_n_batches = sum(node_weights)
        total_batches = epochs * total_n_batches

        node_weights = [weight/total_n_batches for weight in node_weights]        
        model_weights = model.get_weights()

    else:
        results = {"times" : {"train" : [], "epochs" : []}}

        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)

        train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))

        comm.isend(len(train_dataset), dest=0, tag=1000)

        total_batches = epochs * len(train_dataset)
    
    model_weights = comm.bcast(model_weights, root=0)

    if rank != 0:
        model.set_weights(model_weights)

    epoch_start = time.time()
    if rank == 0:
        exited_workers = 0
        latest_tag = 0
        for batch in range(total_batches):

            grads = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

            source = status.Get_source()
            tag = status.Get_tag()

            if latest_tag < tag+1:
                latest_tag = tag+1

            behind_penalty = (tag+1 / latest_tag) #The more behind it is the less impact it will have, verry small penalization

            grads = [grad*node_weights[source-1]*behind_penalty for grad in grads] 

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            comm.send(model.get_weights(), dest=source, tag=tag)

            comm.isend(stop, dest=source, tag=tag)

            if stop:
                exited_workers += 1

            if exited_workers == n_workers:
                break

            if (batch+1) % total_n_batches == 0 and not stop:

                results["times"]["epochs"].append(time.time() - epoch_start)

                print(f"\n End of batch {(batch+1)//n_workers} -> epoch {(batch+1)//total_n_batches}")

                predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
                val_f1 = f1_score(y_cv, predictions, average="macro")
                val_mcc = matthews_corrcoef(y_cv, predictions)
                val_acc = accuracy_score(y_cv, predictions)

                results["acc"].append(val_acc)
                results["f1"].append(val_f1)
                results["mcc"].append(val_mcc)
                results["times"]["global_times"].append(time.time() - start)
                print("- val_f1: %6.3f - val_mcc %6.3f - val_acc %6.3f" %(val_f1, val_mcc, val_acc))
                patience_buffer = patience_buffer[1:]
                patience_buffer.append(val_mcc)

                p_stop = True
                for value in patience_buffer[1:]:
                    if abs(patience_buffer[0] - value) > min_delta:
                        p_stop = False 

                if (val_mcc > early_stop or p_stop) and (batch+1)//total_n_batches > 10:
                    stop = True

                epoch_start = time.time()
            
    else:
        for batch in range(total_batches):

            train_time = time.time()

            x_batch_train, y_batch_train = train_dataset[batch % len(train_dataset)]

            with tf.GradientTape() as tape:

                logits = model(x_batch_train, training=True) 
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            results["times"]["train"].append(time.time() - train_time)

            comm.isend(grads, dest=0, tag=batch)

            model_weights = comm.recv(source=0, tag=batch)

            stop = comm.recv(source=0, tag=batch)

            model.set_weights(model_weights)

            if (batch+1) % len(train_dataset) == 0:
                results["times"]["epochs"].append(time.time() - epoch_start)
                epoch_start = time.time()

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

    