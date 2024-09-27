seed=17
num_workers=2


#python3 data_processing/data_processing.py -d IOT_DNL --seed $seed
#python3 data_processing/data_processing.py -d Slicing5G --seed $seed
#python3 data_processing/data_processing.py -d UNSW --seed $seed
#python3 data_processing/data_processing.py -d NetSlice5G --seed $seed

#python3 data_processing/data_division.py -d IOT_DNL -n $num_workers --seed $seed
#python3 data_processing/data_division.py -d Slicing5G -n $num_workers --seed $seed
#python3 data_processing/data_division.py -d UNSW -n $num_workers --seed $seed
#python3 data_processing/data_division.py -d NetSlice5G -n $num_workers --seed $seed


for i in $(seq 1 $num_workers)
do
    ssh worker$i "cd zenog_fl; source venv/bin/activate; python3 data_processing/data_processing.py -d IOT_DNL --seed $seed"
    ssh worker$i "cd zenog_fl; source venv/bin/activate; python3 data_processing/data_processing.py -d Slicing5G --seed $seed"
    ssh worker$i "cd zenog_fl; source venv/bin/activate; python3 data_processing/data_processing.py -d UNSW --seed $seed"
    ssh worker$i "cd zenog_fl; source venv/bin/activate; python3 data_processing/data_processing.py -d NetSlice5G --seed $seed"

    ssh worker$i "cd zenog_fl; source venv/bin/activate; python3 data_processing/data_division.py -d IOT_DNL -n $num_workers --seed $seed"
    ssh worker$i "cd zenog_fl; source venv/bin/activate; python3 data_processing/data_division.py -d Slicing5G -n $num_workers --seed $seed "
    ssh worker$i "cd zenog_fl; source venv/bin/activate; python3 data_processing/data_division.py -d UNSW -n $num_workers --seed $seed" 
    ssh worker$i "cd zenog_fl; source venv/bin/activate; python3 data_processing/data_division.py -d NetSlice5G -n $num_workers --seed $seed"
done


mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 1 -d Slicing5G -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2
mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 1 -d IOT_DNL -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2
mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 1 -d UNSW -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2
mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 1 -d NetSlice5G -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2

mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 2 -d Slicing5G -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2
mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 2 -d IOT_DNL -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2
mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 2 -d UNSW -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2
mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 2 -d NetSlice5G -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30 -e $seed -a 0.2

mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 3 -d Slicing5G -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed
mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 3 -d IOT_DNL -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed
mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 3 -d UNSW -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed
mpirun -np $((num_workers + 1)) -hostfile MPI/host_file bash run.sh -m 3 -d NetSlice5G -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed

mpirun -np $((num_workers + 1)) -hostfile MPI/host_file -m 4 -d Slicing5G -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed
mpirun -np $((num_workers + 1)) -hostfile MPI/host_file -m 4 -d IOT_DNL -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed
mpirun -np $((num_workers + 1)) -hostfile MPI/host_file -m 4 -d UNSW -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed
mpirun -np $((num_workers + 1)) -hostfile MPI/host_file -m 4 -d NetSlice5G -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30 -e $seed