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

srun --gres=gpu:1 python 

/home/012695917/ME297/Project/Drone-AI-Segmentation/Segmentation/src/train_segnet.py