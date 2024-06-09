mpirun -np 9 -hostfile MPI/host_file python3 federated_learning.py -m 2 -d Slicing5G -ge 300 -le 1 -lr 0.00001 -b 256 -o Adam -s 0.99 -p 30
mpirun -np 9 -hostfile MPI/host_file python3 federated_learning.py -m 2 -d IOT_DNL -ge 300 -le 1 -lr 0.00005 -b 256 -o Adam -s 0.99 -p 30

mpirun -np 9 -hostfile MPI/host_file python3 federated_learning.py -m 1 -d Slicing5G -ge 300 -le 1 -lr 0.00001 -b 256 -a 0.2 -o Adam -s 0.99 -p 30
mpirun -np 9 -hostfile MPI/host_file python3 federated_learning.py -m 1 -d IOT_DNL -ge 300 -le 1 -lr 0.00005 -b 256 -a 0.2 -o Adam -s 0.99 -p 30