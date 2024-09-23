cd /root/SNN/code/Spikingformer-CML/cifar100
CUDA_VISIBLE_DEVICES=2 python train_ablation.py -c config/wsld/vit-s-wsld.yml --temp 1.5