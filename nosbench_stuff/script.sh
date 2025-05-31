#!/bin/bash
#SBATCH --time 0-00:20
#SBATCH --job-name nosbench_array
##SBATCH --partition mlhiwidlc_gpu-rtx2080 
#SBATCH --partition bosch_cpu-cascadelake 
#SBATCH --mem 4000 # memory pool for all cores (4GB)
#SBATCH -c 1 # number of cores
#SBATCH -a 1-1 # array size
##SBATCH --gres=gpu:1  # reserves one GPU
#SBATCH -o log/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e log/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID

# echo "Workingdir: $PWD";
mkdir -p log;

source .venv/bin/activate;

# Variables
OPTIMIZER="RS";
EVALUATIONS=1;
BENCHMARK="NosBench";
# BENCHMARK="ToyBenchmark";
REP_SUFFIX="_rs_100";
FIDELITY=True;

STARTTIME=$(date +%s)

echo "Running nosbench_stuff/nosbench_cluster.py with the following parameters:"
echo "Optimizer: $OPTIMIZER"
echo "Evaluations: $EVALUATIONS"
echo "Benchmark: $BENCHMARK"
echo "Suffix: $REP_SUFFIX"

# Only include -ef argument if FIDELITY is False
if [ "$FIDELITY" = False ]; then
    python nosbench_stuff/nosbench_cluster.py \
        -o $OPTIMIZER \
        -ev $EVALUATIONS \
        -b $BENCHMARK \
        -r $REP_SUFFIX \
        -ef
else
    python nosbench_stuff/nosbench_cluster.py \
        -o $OPTIMIZER \
        -ev $EVALUATIONS \
        -b $BENCHMARK \
        -r $REP_SUFFIX
fi

# Convert elapsed time to hours, minutes, seconds and echo
ELAPSED=$(($(date +%s) - STARTTIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))
# blank line
echo ""
echo "DONE in: $HOURS hours, $MINUTES minutes, $SECONDS seconds"