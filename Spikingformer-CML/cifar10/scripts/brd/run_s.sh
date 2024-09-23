cd /root/SNN/code/Spikingformer-CML/cifar100
python -m torch.distributed.launch --nproc_per_node=8 --master_port='29501' --use_env train.py -c config/brd/vit-s-brd.yml
