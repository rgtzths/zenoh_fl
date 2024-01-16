#!/usr/bin/env bash

set -e


ssh worker1 -C "hostname && exit"
ssh worker2 -C "hostname && exit"
ssh worker3 -C "hostname && exit"
ssh worker4 -C "hostname && exit"

exit 0

case $TEST in 
    mpi)
        sleep infinity
        ;;
    zenoh)
        sleep infinity
        ;;
    *)
        echo "Unknown test $TEST"
        exit -1
        ;;
esac





