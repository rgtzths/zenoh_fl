import json
import pathlib
import time
import pickle

import numpy as np
import tensorflow as tf
import logging

from zcomm import ZCommPy
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG)

async def run(
    dataset_util,
    optimizer,
    early_stop,
    learning_rate,
    batch_size,
    epochs,
    patience,
    min_delta,
    n_workers,
    rank,
    output,
    locator
    ):

    tf.keras.utils.set_random_seed(dataset_util.seed)
    
    best_weights = None
    best_mcc = -1
    stop = False
    dataset = dataset_util.name
    patience_buffer = [-1]*patience

    comm = await ZCommPy.new(rank, n_workers, locator)
    comm.start()
    
    logging.info(f'[RANK: {rank}] Waiting nodes...')
    await comm.wait()
    logging.info(f'[RANK: {rank}] Nodes up!')

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

    start = time.time()

    if rank == 0:
        results = {"acc" : [], "mcc" : [], "f1" : [], "messages_size" : {"sent" : [], "received" : []}, "times" : {"epochs" : [], "global_times" : []}}
        node_weights = [1/n_workers]*n_workers

        X_cv, y_cv = dataset_util.load_validation_data()

        val_dataset = tf.data.Dataset.from_tensor_slices((X_cv, y_cv)).batch(8192)

        total_n_batches = 0
        
        for worker in range(1, n_workers+1):
            data = await comm.recv(src=-2, tag=-10)
            for source, message in data.items():
                total_n_batches += pickle.loads(message.data)
        
    else:

        X_train, y_train = dataset_util.load_worker_data(n_workers, rank)

        train_dataset = list(tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size))

        await comm.send(dest=0, tag=-10, data=pickle.dumps(len(train_dataset)))

        total_n_batches = len(train_dataset)
    
    weights = pickle.loads( (await comm.bcast(data=pickle.dumps(model.get_weights()), root=0, tag=-10)).data)

    if rank != 0:
        model.set_weights(weights)

    if rank == 0:
        exited_workers = 0        
        epoch_start = time.time()
        for batch in range(total_n_batches*epochs):

            data = await comm.recv(src=-2, tag=-2)

            for source, message in data.items():
                grads = pickle.loads(message.data)
                tag = message.tag

                grads = [grad*node_weights[source-1] for grad in grads] 

                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                await comm.send(dest=source, tag=tag, data=pickle.dumps(model.get_weights()))
                await comm.send(dest=source, tag=tag, data=pickle.dumps(stop))

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

            await comm.send(data=pickle.dumps(grads), dest=0, tag=batch)

            data = await comm.recv(src=0, tag=batch)
            for src, message in data.items():
                weights = pickle.loads(message.data)

            data = await comm.recv(src=0, tag=batch)
            for src, message in data.items():
                stop = pickle.loads(message.data)

            model.set_weights(weights)

            if stop:
                break

    if rank==0:
        history = json.dumps(results)
        model.set_weights(best_weights)
        model.save(output/'centralized_async.keras')
        with open(output/"server.json", "w") as f:
            f.write(history)

    
