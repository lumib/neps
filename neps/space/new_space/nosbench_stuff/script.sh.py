#!/bin/bash
#SBATCH --time 0-00:20
#SBATCH --job-name nosbench_array
##SBATCH --partition mlhiwidlc_gpu-rtx2080 
#SBATCH --partition bosch_cpu-cascadelake 
#SBATCH --mem 4000 # memory pool for all cores (4GB)
#SBATCH -c 1 # number of cores
#SBATCH -a 1-10 # array size
##SBATCH --gres=gpu:1  # reserves one GPU
#SBATCH -o log/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e log/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID

echo "Workingdir: $PWD";
# rm -rf log;
mkdir -p log;

source .venv/bin/activate;

# Plugin your python script here
python neps/space/new_space/nosbench_stuff/nosbench_cluster.py;
echo "done"