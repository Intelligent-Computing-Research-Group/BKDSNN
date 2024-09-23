cd /root/SNN/code/Spikingformer-CML/cifar10
CUDA_VISIBLE_DEVICES=4 python train_ablation.py -c config/brd/vit-s-brd.yml --lambda_mgd 0.6
