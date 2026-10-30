#!/bin/bash
#SBATCH --job-name=mas-feas-e01
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --nodelist=e01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/snapaug1/multiagent_feasibility/pilot/logs/gpu_job_e01_%j.out
sleep 28800
