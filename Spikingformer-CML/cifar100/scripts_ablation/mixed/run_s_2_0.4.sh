cd /root/SNN/code/Spikingformer-CML/cifar100
CUDA_VISIBLE_DEVICES=0 python train_ablation.py -c config/mixed/vit-s-mixed.yml --lambda_mgd 0.4