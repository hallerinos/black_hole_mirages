#!/bin/bash
module load lang/Anaconda3/2020.11
CONDA_HOME=/opt/apps/resif/iris/2020b/broadwell/software/Anaconda3/2020.11
. $CONDA_HOME/etc/profile.d/conda.sh
conda create --name kwant_env
conda activate kwant_env
conda install -c conda-forge kwant
conda info
# conda deactivate