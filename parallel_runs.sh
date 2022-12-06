#!/bin/bash
module load lang/Anaconda3/2020.11
# CONDA_HOME=/opt/apps/resif/iris/2020b/broadwell/software/Anaconda3/2020.11  # use this on iris
CONDA_HOME=/opt/apps/resif/aion/2020b/epyc/software/Anaconda3/2020.11  # use this on aion
. $CONDA_HOME/etc/profile.d/conda.sh
conda activate kwant_env

MAIN="python codes/lattice_simulation.py"
CFG_PATH=$1

N_PARA=1  # threads/task

N_CPUS=$(nproc)
N_JOBS=$(($N_CPUS/$N_PARA))  # how many tasks in parallel

export OMP_NUM_THREADS=1  # KWANT does not allow multi-threading
export SLURM_JOB_ID=1337  # just to have a job id (for checkdone.jl)

echo "#CPU(s): $N_CPUS, #threads/job: $OMP_NUM_THREADS, #tasks in parallel: $N_JOBS"

FILE=stop
if test -f "$FILE"; then
   echo "remove stop file"
   rm $FILE
fi

JOBS_TOTAL=$(find . -path "*$CFG_PATH/*.json" | wc -l)
N_BDLES=$(($JOBS_TOTAL/$N_JOBS + 1))

echo "Total number of requested jobs in $CFG_PATH: $JOBS_TOTAL"

bdl=0
ctr=0
for cfg in $CFG_PATH/*.json; do
   # skip if empty
   [ -e "$cfg" ] || continue
   
   # increase counter
   ctr=$(($ctr + 1))
   
   # set up task
   CMD="$MAIN $cfg"
   
   # run in parallel
   $CMD &

   # wait if socket is full
   if [ $ctr -eq $N_JOBS ]; then
      echo "Waiting for job bundle $bdl/$N_BDLES to finish."
      wait
      ctr=0
      bdl=$(($bdl + 1))
   fi
done
wait