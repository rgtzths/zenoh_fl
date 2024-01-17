#!/usr/bin/env bash

set -e


case $TEST in 
    mpi)
        if [[ ! -z ${MASTER} ]]; then
            echo "Master starting..."
            ssh worker1 -C "hostname && exit"
            ssh worker2 -C "hostname && exit"
            ssh worker3 -C "hostname && exit"
            ssh worker4 -C "hostname && exit"

            mpirun -np 5 -hostfile host_file python3.10 federated_learning.py -m 1 -d IOT_DNL -lr 0.000005 -b 1024 -s 0.96 -c mpi
            sleep 10
            mpirun -np 5 -hostfile host_file python3.10 federated_learning.py -m 2 -d IOT_DNL -lr 0.000005 -b 1024 -s 0.96 -c mpi
            sleep 10
            mpirun -np 5 -hostfile host_file python3.10 federated_learning.py -m 3 -d IOT_DNL -lr 0.000005 -b 1024 -s 0.96 -ge 300 -c mpi
            sleep 10
            mpirun -np 5 -hostfile host_file python3.10 federated_learning.py -m 3 -d IOT_DNL -lr 0.000005 -b 1024 -s 0.96 -ge 300 -c mpi
        else
            /usr/sbin/sshd -D
        fi
        ;;
    zenoh)
        sleep infinity
        ;;
    *)
        echo "Unknown test $TEST"
        exit -1
        ;;
esac





