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
        echo "Running centarlized async"
        python3 z_centralized_async.py -d  dataset/one_hot_encoding -o /results/zenoh-centralized-async -e $EPOCHS -w 3 -r $RANK
        touch /results/zenoh-centralized-async/done
        ;;
    sync_decentralized)
        echo "Running decentarlized sync"
        python3 z_decentralized_sync.py -d  dataset/one_hot_encoding  -o /results/zenoh-decentralized-sync -w 3 -r $RANK -l 0.1 # $EPOCHS
        touch /results/zenoh-decentralized-sync/done
        ;;
    async_decentralized)
        echo "Running decentarlized async"
        python3 z_decentralized_async.py -d dataset/one_hot_encoding -o /results/zenoh-decentralized-async -w 3 -r $RANK -l 0.1 #-e $EPOCHS
        touch /results/zenoh-decentralized-async/done
        ;;
    *)
        echo "Unknown test $TEST"
        exit -1
        ;;
esac





