cd /root/SNN/code/Spikingformer-CML/imagenet
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' --use_env train.py -c config/mixed/vit-m-mixed.yml
