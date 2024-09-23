#!/bin/bash

#SBATCH --job-name=test
#SBATCH --partition=debuga100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=25
#SBATCH --gres=gpu:2           #最多28gpu卡
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load gcc cuda
module load miniconda3
source ~/env/snn/bin/activate

cd /dssg/home/acct-seehzz/seehzz-xzk/SNN/code/Spike-Element-Wise-ResNet/imagenet
python -m torch.distributed.launch --nproc_per_node=2 --use_env train_distillation.py \
                                                           --cos_lr_T 320 \
                                                           --model sew_resnet50 -b 128 \
                                                           --output-dir /dssg/home/acct-seehzz/seehzz-xzk/SNN/code/Spike-Element-Wise-ResNet/imagenet/logs --tb --print-freq 64 \
                                                           --amp --cache-dataset --connect_f ADD --T 4 \
                                                           --lr 0.1 --epoch 320 --data-path /dssg/home/acct-seehzz/seehzz-xzk/dataset/ImageNet
