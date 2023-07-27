#!/bin/bash
#SBATCH --mem=16g
#SBATCH --time=1-0
#SBATCH --gres=gpu:1,vmem:20g
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --exclude=ampere-01,arion-01,arion-01,arion-02,binky-01,binky-02,binky-03,binky-04,binky-05

sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
module load cuda
module load torch
export TRANSFORMERS_CACHE=/cs/snapless/gabis/eliyahabba/.cache/huggingface/transformers
export HF_HOME=/cs/snapless/gabis/eliyahabba/.cache/huggingface/
export HF_DATASETS_CACHE=/cs/snapless/gabis/eliyahabba/.cache/huggingface/datasets

dir=/cs/snapless/gabis/eliyahabba/Legal_nlp/SoundOfAttention

cd $dir

source /cs/snapless/gabis/eliyahabba/Legal_nlp/venvs/soundVenv/bin/activate



# the SLURM_ARRAY_TASK_ID variable will get the appropiate array index. using the dollar sign is the way to refrence variables in bash (and echo just prints).
echo ${SLURM_ARRAY_TASK_ID}
CUDA_LAUNCH_BLOCKING=1 python AttentionsAnalysis/AttentionsDataCreator.py
