#!/usr/bin/env bash

set -e


ssh worker1 -C "hostname && exit"
ssh worker2 -C "hostname && exit"
ssh worker3 -C "hostname && exit"
cd code


case $TEST in 
    sync_centralized)
        echo "Running centarlized sync"
        mpirun -np 4 -hostfile hostfile python mpi_centralized_sync.py -d dataset -o /results/mpi-centralized-sync -e $EPOCHS
        touch /results/mpi-centralized-sync/done
        ;;
    async_centralized)
        echo "Running centarlized async"
        mpirun -np 4 -hostfile hostfile python mpi_centralized_assync.py -d dataset -o /results/mpi-centralized-async -e $EPOCHS
        touch /results/mpi-centralized-async/done
        ;;
    sync_decentralized)
        echo "Running decentarlized sync"
        mpirun -np 4 -hostfile hostfile python mpi_decentralized_sync.py -d dataset -o /results/mpi-decentralized-sync # $EPOCHS
        touch /results/mpi-decentralized-sync/done
        ;;
    async_decentralized)
        echo "Running decentarlized async"
        mpirun -np 4 -hostfile hostfile python mpi_decentralized_assync.py -d dataset -o /results/mpi-decentralized-async #-e $EPOCHS
        touch /results/mpi-decentralized-async/done
        ;;
    *)
        echo "Unknown test $TEST"
        exit -1
        ;;
esac





