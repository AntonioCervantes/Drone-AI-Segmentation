#!/bin/bash
#
#SBATCH --job-name=AC_segnet
#SBATCH --output=training.log
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mail-user=antonio.cervantes@sjsu.edu
#SBATCH --mail-type=END
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

source /home/012695917/anaconda3/etc/profile.d/conda.sh
conda activate tf2
module load cuda/10.1
srun --gres=gpu:1 python 

python /home/012695917/ME297/Project/Drone-AI-Segmentation/Segmentation/src/train_segnet.py