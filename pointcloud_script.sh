#!/bin/bash

N=14
PIDS=()

cleanup() {
    echo
    echo "Stopping all point-cloud processes..."

    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done

    echo "All processes stopped."
    exit 1
}

trap cleanup SIGINT SIGTERM

for ((i=0; i<N; i++))
do
    echo "Starting process $i"

    python pointcloud_phase.py "$i" "$N" &

    PIDS+=($!)

    sleep 1
done

wait

echo "All point cloud processes finished."
