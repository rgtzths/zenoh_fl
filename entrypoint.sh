#!/usr/bin/env bash

set -e
if [[ -z ${TEST} ]]; then
    echo "Missing TEST environment variable, cannot continue!"
    exit -1
fi 

if [[ -z ${COMM} ]]; then
    echo "Missing COMM environment variable, cannot continue!"
    exit -1
fi 


case $COMM in 
    mpi)
        if [[ ! -z ${MASTER} ]]; then
            echo "Master starting..."
            ssh worker1 -C "hostname && exit"
            ssh worker2 -C "hostname && exit"
            ssh worker3 -C "hostname && exit"
            ssh worker4 -C "hostname && exit"

            case $TEST in
                1)  
                    mpirun -np 5 -hostfile host_file python3.10 federated_learning.py -m 1 -d IOT_DNL -lr 0.000005 -b 1024 -s 0.96 -c mpi
                    sleep 10
                    ;;
                2)
                    mpirun -np 5 -hostfile host_file python3.10 federated_learning.py -m 2 -d IOT_DNL -lr 0.000005 -b 1024 -s 0.96 -c mpi
                    sleep 10
                    ;;
                3)  
                    mpirun -np 5 -hostfile host_file python3.10 federated_learning.py -m 3 -d IOT_DNL -lr 0.000005 -b 1024 -s 0.96 -ge 300 -c mpi
                    sleep 10
                    ;;
                4)
                    mpirun -np 5 -hostfile host_file python3.10 federated_learning.py -m 3 -d IOT_DNL -lr 0.000005 -b 1024 -s 0.96 -ge 300 -c mpi
                    sleep 10
                    ;;
            esac

        else
            /usr/sbin/sshd -D
        fi
        ;;
    zenoh)
        
        case $TEST in
            1)
                python3.10 federated_learning.py -m 1 -d IOT_DNL -lr 0.000005 -b 1024 -s 0.96 -c zenoh -nw 4 -wid $RANK
                sleep 10
                ;;
            2)
                python3.10 federated_learning.py -m 2 -d IOT_DNL -lr 0.000005 -b 1024 -s 0.96 -c zenoh -nw 4 -wid $RANK
                sleep 10
                ;;
            3)  
                python3.10 federated_learning.py -m 3 -d IOT_DNL -lr 0.000005 -b 1024 -s 0.96 -ge 300 -c zenoh -nw 4 -wid $RANK
                sleep 10
                ;;
            4)  
                python3.10 federated_learning.py -m 3 -d IOT_DNL -lr 0.000005 -b 1024 -s 0.96 -ge 300 -c zenoh -nw 4 -wid $RANK
                sleep 10
                ;;
        esac
        ;;
    *)
        echo "Unknown test $TEST"
        exit -1
        ;;
esac





