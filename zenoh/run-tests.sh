#!/usr/bin/env bash

set -e 


plog () {
   LOG_TS=`eval date "+%F-%T"`
   echo "[$LOG_TS]: $1"
}

usage() { printf "Usage: $0 \n\t
   -a async centralized\n\t
   -S sync decentralized\n\t
   -A async decentralized\n\t
   -s sync centralized\n\t
   -h help\n" 1>&2; exit 1; }

# trap ctrl-c and call ctrl_c()
trap ctrl_c INT

function ctrl_c() {
    cleanup
    exit
}


# kills all the processes
function cleanup() {
   kill -9 $COMPOSE_PID > /dev/null 2>&1
}



while getopts "aAsSh" arg; do
   case ${arg} in
   h)
      usage
      ;;
   s)
        plog "[RUN] Zenoh Sync centralized"
        export TEST=sync_centralized
        bash -c "docker compose up" &
        COMPOSE_PID=$!


        while [ ! -f results/zenoh-centralized-sync/done ]
        do
        sleep 20
        done
        ls -l results/zenoh-centralized-sync/

        kill -2 $COMPOSE_PID
        docker compose down
        unset $TEST
        plog "[DONE] Zenoh Sync centralized"
        ;;
    S)
        plog "[RUN] Zenoh Sync decentralized"
        export TEST=sync_decentralized
        bash -c "docker compose up" &
        COMPOSE_PID=$!


        while [ ! -f results/zenoh-decentralized-sync/done ]
        do
        sleep 20
        done
        ls -l results/zenoh-decentralized-sync/

        kill -2 $COMPOSE_PID
        docker compose down
        unset $TEST
        plog "[DONE] Zenoh Sync decentralized"
        ;;
    a)
        plog "[RUN] Zenoh async centralized"
        export TEST=async_centralized
        bash -c "docker compose up" &
        COMPOSE_PID=$!

        while [ ! -f results/zenoh-centralized-async/done ]
        do
        sleep 20
        done
        ls -l results/zenoh-centralized-async/

        kill -2 $COMPOSE_PID
        docker compose down
        unset $TEST
        plog "[DONE] Zenoh async centralized"
        ;;
    A)
        plog "[RUN] Zenoh async decentralized"
        export TEST=async_decentralized
        bash -c "docker compose up" &
        COMPOSE_PID=$!


        while [ ! -f results/zenoh-decentralized-async/done ]
        do
        sleep 20
        done
        ls -l results/zenoh-decentralized-async/

        kill -2 $COMPOSE_PID
        docker compose down
        unset $TEST
        plog "[DONE] Zenoh async decentralized"
        ;;
    *)
      usage
      ;;
   esac
done

cleanup
plog "[ DONE ] Bye!"