#!/bin/bash
#SBATCH --time 1-00:00
#SBATCH --job-name nosbench
##SBATCH --partition mlhiwidlc_gpu-rtx2080 
#SBATCH --partition bosch_cpu-cascadelake
#SBATCH --mem 4000 # memory pool for all cores (4GB)
#SBATCH -c 1 # number of cores
#SBATCH -a 1-1 # array size
##SBATCH --gres=gpu:1  # reserves one GPU
#SBATCH -o log/%x.%A/out/%N.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e log/%x.%A/err/%N.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID

mkdir -p log;
source .venv/bin/activate;

# Variables
OPTIMIZER="PB+ASHB";
# EVALUATIONS=25000;
COST=25;
DIR_NAME="test_warmstart";
WARMSTART="AdamW";
BENCHMARK="NosBench";
# BENCHMARK="ToyBenchmark";

FIDELITY=True;

STARTTIME=$(date +%s)

echo "Running nosbench_stuff/nosbench_cluster.py with the following parameters:"
echo "Optimizer: $OPTIMIZER"
if [[ -v EVALUATIONS ]]; then
    echo "Evaluations: $EVALUATIONS"
else
    echo "Cost: $COST"
fi
if [[ -v WARMSTART ]]; then
    echo "Warmstart: $WARMSTART"
else
    echo "No warmstart"
fi
echo "Benchmark: $BENCHMARK"
echo "Name: $DIR_NAME"

# Initialize an array for arguments
cmd_args=(
    -o "$OPTIMIZER"
    -b "$BENCHMARK"
    -d "$DIR_NAME"
)

# Add arguments based on EVALUATIONS
if [[ -v EVALUATIONS ]]; then
    cmd_args+=(-ev "$EVALUATIONS") # Add -ev if EVALUATIONS is empty/unset
else
    cmd_args+=(-mct "$COST")       # Add -mct if EVALUATIONS is not empty
fi

# Add warmstart argument if WARMSTART is set
if [[ WARMSTART ]]; then
    cmd_args+=(-ws "$WARMSTART")   # Add -ws if WARMSTART is set
fi

# Add flag based on FIDELITY
if [ "$FIDELITY" = False ]; then
    cmd_args+=(-ef)                # Add -ef if FIDELITY is False
fi
echo ${cmd_args[@]} # Print the constructed command arguments for debugging
# Execute the python script with the constructed arguments
python nosbench_stuff/nosbench_cluster.py "${cmd_args[@]}"



# Convert elapsed time to hours, minutes, seconds and echo
ELAPSED=$(($(date +%s) - STARTTIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))
# blank line
echo ""
echo "DONE in: $HOURS hours, $MINUTES minutes, $SECONDS seconds"