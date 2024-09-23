cd /root/SNN/code/Spikingformer-CML/cifar100
CUDA_VISIBLE_DEVICES=1 python train_ablation.py -c config/brd/vit-s-brd.yml --lambda_mgd 0.4
