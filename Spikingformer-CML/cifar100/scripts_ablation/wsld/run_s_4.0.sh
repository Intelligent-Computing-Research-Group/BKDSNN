cd /root/SNN/code/Spikingformer-CML/cifar100
CUDA_VISIBLE_DEVICES=5 python train_ablation.py -c config/wsld/vit-s-wsld.yml --temp 4.0