#!/bin/

source venv/bin/activate

seed=17
num_workers=33

datasets=("IOT_DNL" "Slicing5G" "UNSW" "NetSlice5G")

for i in $(seq 1 $num_workers)
do
    ssh worker$i "pkill -f 'python3 federated_learning.py'"
done

#for i in $(seq 1 $num_workers)
#do
#    ssh-copy-id worker$i
#done

#To fix any mistakes without having to Push pull gits
#for i in $(seq 1 $num_workers)
#do
#    ssh worker$i "cd zenoh_fl; rm -r ZENOH; rm federated_learning.py; rm -r MPI;"
#    scp -r ZENOH atnoguser@worker$i:~/zenoh_fl/
#    scp -r MPI atnoguser@worker$i:~/zenoh_fl/
#    scp federated_learning.py atnoguser@worker$i:~/zenoh_fl/
#done


#Code running
#for dataset in "${datasets[@]}"; do
#    python3 data_processing/data_processing.py -d "$dataset" --seed "$seed"
#    python3 data_processing/data_division.py -d "$dataset" -n $num_workers --seed $seed
#done
#
#for i in $(seq 1 $num_workers)
#do
#    echo worker$i
#    for dataset in "${datasets[@]}"; do
#        ssh worker$i "cd zenoh_fl; rm -r datasets/${dataset}/data/${seed}/${num_workers}_workers"
#        scp -r datasets/"$dataset"/data/$seed/"$num_workers"_workers atnoguser@worker$i:~/zenoh_fl/datasets/"$dataset"/data/$seed
#    done
#done

#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 1 -d Slicing5G -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 1 -d IOT_DNL -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 1 -d UNSW -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 1 -d NetSlice5G -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2
#
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 2 -d Slicing5G -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 2 -d IOT_DNL -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 2 -d UNSW -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 2 -d NetSlice5G -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2
#
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 3 -d Slicing5G -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 3 -d IOT_DNL -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 3 -d UNSW -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 3 -d NetSlice5G -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed
#
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 4 -d Slicing5G -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 4 -d IOT_DNL -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 4 -d UNSW -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed
#mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 4 -d NetSlice5G -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed

# Command to run
# Loop through each host and execute the command
#for i in $(seq 1 $num_workers)
#do
#    ssh worker$i "cd zenoh_fl; source venv/bin/activate; nohup python3 federated_learning.py -m 1 -d Slicing5G -lr 0.000005 -ge 300 -lr 0.00005 -le 1 -o Adam -b 256 -s 0.99 -p 30 -a 0.2 -c zenoh -nw ${num_workers} -wid ${i} --seed ${seed} > output.log 2>&1 &"
#done
#python3 federated_learning.py -m 1 -d Slicing5G -lr 0.000005 -ge 300 -lr 0.00005 -le 1 -o Adam -b 256 -s 0.99 -p 30 -a 0.2 -c zenoh -nw $num_workers -wid 0 --seed ${seed}