#!/bin/bash

#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -p sched_mit_sloan_interactive
#SBATCH --time=3-11:55
#SBATCH -o notebook_%A.out

IP=`hostname -i`
PORT=`shuf -i 2000-65000 -n 1`

# module load sloan/python/modules/3.5

# $HOME/.local/bin/jupyter notebook --ip=$IP --port=$PORT --no-browser

$HOME/anaconda3/bin/jupyter notebook --ip=$IP --port=$PORT --no-browser
