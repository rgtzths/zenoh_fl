#!/bin/bash

source venv/bin/activate

seed=42
num_workers=37

datasets=("IOT_DNL" "Slicing5G" "UNSW" "NetSlice5G")
#datasets=("UNSW" "NetSlice5G")

##Helper commands for setting up for the first time

#Ensures every worker can be accessed with priv/pub key
#for i in $(seq 1 $num_workers)
#do
#    ssh-copy-id worker$i
#done

#Kills any program lauched using the federated_learning script (for Zenoh)
for i in $(seq 1 $num_workers)
do
    ssh worker$i "pkill -f 'python3 federated_learning.py'"
    #ssh worker$i "pkill -f 'python3 zcomm/example.py'"
done

#Sends the federated learning scripts (both zenoh and mpi)
#for i in $(seq 1 $num_workers)
#do
#    ssh worker$i "cd zenoh_fl; rm -r ZENOH; rm federated_learning.py; rm -r MPI;"
#    scp -r ZENOH atnoguser@worker$i:~/zenoh_fl/
#    scp -r MPI atnoguser@worker$i:~/zenoh_fl/
#    scp federated_learning.py atnoguser@worker$i:~/zenoh_fl/
#done

#Code running

## Runs datasets preprocessing and division
#for dataset in "${datasets[@]}"; do
#    python3 data_processing/data_processing.py -d "$dataset" --seed "$seed"
#    python3 data_processing/data_division.py -d "$dataset" -n $num_workers --seed $seed
#done
#
### Copies the datasets to the workers
#for i in $(seq 1 $num_workers)
#do
#    for dataset in "${datasets[@]}"; do
#        ssh worker$i "cd zenoh_fl; rm -r datasets/${dataset}/data/${seed}/${num_workers}_workers"
#        scp -r datasets/"$dataset"/data/$seed/ atnoguser@worker$i:~/zenoh_fl/datasets/"$dataset"/data/
#    done
#done

#Running MPI
#echo "Running experiments"
for dataset in "${datasets[@]}"; do
    for m in $(seq 3 4)
    do
        mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m $m -d $dataset -g 300 -l 3 -r 0.001 -b 256 -o Adam -s 0.99 -p 25 -e $seed -a 0.2
    done
done

#Running Zenoh
for dataset in "${datasets[@]}"; do
    for m in $(seq 3 4)
    do
        for i in $(seq 1 $num_workers)
        do
            ssh worker$i "cd zenoh_fl; source venv/bin/activate; nohup python3 federated_learning.py -m $m -d $dataset -lr 0.001 -ge 300 -le 3 -o Adam -b 256 -s 0.99 -p 25 -a 0.2 -c zenoh -nw ${num_workers} -wid ${i} --seed ${seed} > output.log 2>&1 &"
        done
        python3 federated_learning.py -m $m -d $dataset -lr 0.001 -ge 300 -le 3 -o Adam -b 256 -s 0.99 -p 25 -a 0.2 -c zenoh -nw $num_workers -wid 0 --seed ${seed}
        for i in $(seq 1 $num_workers)
        do
            ssh worker$i "pkill -f 'python3 federated_learning.py'"
            #ssh worker$i "pkill -f 'python3 zcomm/example.py'"
        done
    done
done
