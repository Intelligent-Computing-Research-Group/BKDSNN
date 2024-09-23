cd /root/SNN/code/Spikingformer-CML/cifar100
python -m torch.distributed.launch --nproc_per_node=8 --master_port='29500' --use_env train.py -c config/mixed/vit-ti-mixed.yml
