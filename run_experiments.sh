#!/bin/

source venv/bin/activate

seed=17
num_workers=2

datasets=("IOT_DNL" "Slicing5G" "UNSW" "NetSlice5G")


#To fix any mistakes without having to Push pull gits
for i in $(seq 1 $num_workers)
do
    ssh worker$i "cd zenoh_fl; rm -r ZENOH"
    scp -r ZENOH atnoguser@worker$i:~/zenoh_fl/
done


#Code running
#for dataset in "${datasets[@]}"; do
#    python3 data_processing/data_processing.py -d "$dataset" --seed "$seed"
#    python3 data_processing/data_division.py -d "$dataset" -n $num_workers --seed $seed
#done


#for i in $(seq 1 $num_workers)
#do
#    for dataset in "${datasets[@]}"; do
#        ssh worker$i "cd zenoh_fl; rm -r datasets/${dataset}/data/${seed}"
#        scp -r datasets/"$dataset"/data/$seed atnoguser@worker$i:~/zenoh_fl/datasets/"$dataset"/data/
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