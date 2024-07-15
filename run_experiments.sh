#mpirun -np 9 -hostfile MPI/host_file bash run.sh -m 2 -d Slicing5G -g 300 -l 1 -r 0.00001 -b 256 -o Adam -s 0.99 -p 30
#mpirun -np 9 -hostfile MPI/host_file bash run.sh -m 2 -d IOT_DNL -g 300 -l 1 -r 0.00005 -b 256 -o Adam -s 0.99 -p 30

mpirun -np 9 -hostfile MPI/host_file bash run.sh -m 1 -d Slicing5G -g 300 -l 1 -r 0.00001 -b 256 -a 0.2 -o Adam -s 0.99 -p 30
mpirun -np 9 -hostfile MPI/host_file bash run.sh -m 1 -d IOT_DNL -g 300 -l 1 -r 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30