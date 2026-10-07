#!/bin/bash

# Usage:
#   ./pointcloud_script.sh N                  # N workers, ALL runs in ASSET_DIR
#   ./pointcloud_script.sh N 14               # only run 14 (matches run_0014.h5 or run_14.h5)
#   ./pointcloud_script.sh N 14 22 87         # several specific runs
#   ./pointcloud_script.sh N 20-30            # an inclusive range (missing runs skipped)
#   ./pointcloud_script.sh N 14 20-30 87      # any mix
#
# Everything printed (including Python worker output and errors) is shown on
# screen AND saved to a timestamped log file: log_YYYYMMDD_HHMMSS.txt

N=${1:-1}
shift 2>/dev/null
SELECTED=("$@")

if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
    echo "First argument must be a positive number of workers (got '$N')" >&2
    exit 2
fi

# ---------------------------------
# Logging: copy stdout + stderr to a log file (and keep printing to screen)
# ---------------------------------
LOG_FILE="log_$(date +%Y%m%d_%H%M%S).txt"
# To use one fixed file that keeps appending, use instead:
# LOG_FILE="log.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to $LOG_FILE"
echo "Started: $(date)"
echo "Command: $0 $*"

# Restrict threaded libraries
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

format_time() {
    local seconds=$1
    printf '%02d:%02d:%02d' \
        $((seconds / 3600)) \
        $(((seconds % 3600) / 60)) \
        $((seconds % 60))
}


ASSET_DIR="/mnt/analysis/argonne2023/a2066/h5"

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

# Find the h5 file for a run number, accepting 14 or 0014
find_run_file() {
    local r=$1 cand f
    [[ "$r" =~ ^[0-9]+$ ]] || return 1
    for cand in "$r" "$(printf '%04d' "$((10#$r))")"; do
        f="$ASSET_DIR/run_${cand}.h5"
        if [ -e "$f" ]; then
            echo "$f"
            return 0
        fi
    done
    return 1
}

# ---------------------------------
# Build the list of files to process
# ---------------------------------
FILES=()

if [ ${#SELECTED[@]} -eq 0 ]; then
    for file in "$ASSET_DIR"/run_00*.h5; do
        [ -e "$file" ] || continue
        FILES+=("$file")
    done
else
    for arg in "${SELECTED[@]}"; do
        if [[ "$arg" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            lo=$((10#${BASH_REMATCH[1]}))
            hi=$((10#${BASH_REMATCH[2]}))
            for ((r=lo; r<=hi; r++)); do
                if f=$(find_run_file "$r"); then
                    FILES+=("$f")
                fi
            done
        elif f=$(find_run_file "$arg"); then
            FILES+=("$f")
        else
            echo "WARNING: no file found for run '$arg' in $ASSET_DIR" >&2
        fi
    done
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No runs to process." >&2
    exit 1
fi

echo "Processing ${#FILES[@]} run(s) with $N workers each."

FAILED_RUNS=()

# ---------------------------------
# Point-cloud phase
# ---------------------------------
for file in "${FILES[@]}"; do

    run=$(basename "$file" | sed -E 's/^run_(.*)\.h5$/\1/')

    echo "======================================"
    echo "Starting run $run"
    echo "File: $file"
    echo "======================================"

    # Start timer for this run
    SECONDS=0

    PIDS=()

    for ((i=0; i<N; i++)); do
        # -u = unbuffered, so prints reach the log immediately
        python -u pointcloud_phase.py "$i" "$N" "$run" &
        PIDS+=($!)
    done

    # Wait on each worker individually so we can see its exit code
    fail=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || fail=1
    done

    if [ "$fail" -ne 0 ]; then
        echo "Run $run: one or more workers FAILED, skipping compile."
        echo "Run $run took ${SECONDS}s before failure."
        FAILED_RUNS+=("$run")
        continue
    fi

    echo "All point-cloud processes finished for run $run."
    echo "Point-cloud phase took ${SECONDS}s."

    # Compile this run
    compile_start=$SECONDS

    if ! python -u pointcloud_compile.py "$run"; then
        echo "Run $run: compile step FAILED."
        echo "Run $run took ${SECONDS}s total."
        FAILED_RUNS+=("$run")
        continue
    fi

    compile_time=$((SECONDS - compile_start))
    total_time=$SECONDS

    echo "Compile took ${compile_time}s."
    echo "Finished run $run in ${total_time}s."
done

echo "======================================"
echo "Ended: $(date)"
if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo "All runs finished successfully."
else
    echo "Finished, but these runs failed: ${FAILED_RUNS[*]}" >&2
    exit 1
fi
