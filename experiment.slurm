#!/bin/bash
#SBATCH --job-name=ptr_arch_comp        # create a short name for your job
#SBATCH --partition=kempner             # partition
#SBATCH --account=kempner_bsabatini_lab # account needed for kempner partition
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks-per-node=1             # total number of tasks per node
#SBATCH --cpus-per-task=64              # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1                    # number of allocated gpus per node
#SBATCH --mem=64G                     # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=08:00:00                 # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin               # send email when job begins
#SBATCH --mail-type=end                 # send email when job ends

# we need to define the job name directly since it isn't a slurm environment variable
JOB_NAME="ptr_arch_comp" 

# this text file has some settings in it (like the standard job directory)
source cluster/slurm_settings.txt
JOB_DIR=${JOB_FOLDER}/${JOB_NAME}-${SLURM_JOB_ID} # make a specific directory for this particular job
mkdir -p $JOB_DIR

# define a unique log file in the right place
logfile="${JOB_DIR}/slurm-${SLURM_JOB_ID}.out"
echo "Writing to ${logfile}"

# load python and activate our conda environment
module purge
module load python
source activate pointersequencer

# record the start time
start_time=$(date +%s)

# Run the experiment
python experiment.py ptr_arch_comp --task dominoe_sequencer --train_epochs 2000 --replicates 3 --encoder_method attention --decoder_method attention --save_ckpts >> $logfile

# record the end time
end_time=$(date +%s)

# measure the time elapsed for the core part of the job in the logfile
total_time=$((end_time-start_time))
echo "Total Time= "$total_time" seconds" >> $logfile