#!/bin/bash
#SBATCH -p gpu-limited                       # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 
#SBATCH --ntasks-per-node=1		# Specify number of tasks per node
#SBATCH --gpus-per-node=1		        # Specify total number of GPUs
#SBATCH -t 24:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200258                       # Specify project name
#SBATCH -J evaluate                       # Specify job name
#SBATCH --output=./logs/thaillm8b-sft-1.out                 # Specify output file

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn 
#export FI_MR_CACHE_MONITOR=memhooks
#export NCCL_NET_GDR_LEVEL=3
#export NCCL_NET=IB
#export NCCL_IB_HCA=mlx5
#export CXI_FORK_SAFE=1 
#export CXI_FORK_SAFE_HP=1 
#export FI_CXI_DISABLE_CQ_HUGETLB=1

#echo "using FI_MR_CACHE_MONITOR=memhooks"

START=`date`
starttime=$(date +%s)

export WANDB_MODE="offline"

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 6000-9999 -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export PYTHONNOUSERSITE=1
echo go $COUNT_NODE
echo $HOSTNAMES

module restore
module load Mamba
module load cpe-cuda/23.09
module load cudatoolkit/24.11_12.6
conda activate /project/lt200258-aithai/llm/env-list/cuda12_6_torch2_8

if [ -z "$1" ]; then
  echo "Error: Checkpoint path argument missing"
  exit 1
fi

CHECKPOINT_PATH=$1

python -m src.run --model-path "${CHECKPOINT_PATH}" "${@:2}"
END=`date`
endtime=$(date +%s)
echo "Job start at" $START
echo "Job end   at" $END
