#!/usr/bin/env bash

set -e

cd code


case $TEST in 
    sync_centralized)
        echo "Running centarlized sync"
        python3 z_centralized_sync.py -d dataset/one_hot_encoding -o /results/zenoh-centralized-sync -e $EPOCHS -w 3 -r $RANK
        touch /results/zenoh-centralized-sync/done
        ;;
    async_centralized)
        echo "Not yet..."
        # echo "Running centarlized async"
        # mpirun -np 4 -hostfile hostfile python mpi_centralized_assync.py -d dataset -o /results/mpi-centralized-async -e $EPOCHS
        # touch /results/mpi-centralized-async/done
        ;;
    sync_decentralized)
        echo "Not yet..."
        # echo "Running decentarlized sync"
        # mpirun -np 4 -hostfile hostfile python mpi_decentralized_sync.py -d dataset -o /results/mpi-decentralized-sync # $EPOCHS
        # touch /results/mpi-decentralized-sync/done
        ;;
    async_decentralized)
        echo "Not yet..."
        # echo "Running decentarlized async"
        # mpirun -np 4 -hostfile hostfile python mpi_decentralized_assync.py -d dataset -o /results/mpi-decentralized-async #-e $EPOCHS
        # touch /results/mpi-decentralized-async/done
        ;;
    *)
        echo "Unknown test $TEST"
        exit -1
        ;;
esac





