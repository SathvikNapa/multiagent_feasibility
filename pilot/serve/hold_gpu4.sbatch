#!/bin/bash
#SBATCH --job-name=mas-feas-pilot4
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/snapaug1/multiagent_feasibility/pilot/logs/gpu_job4_%j.out
sleep 172800
