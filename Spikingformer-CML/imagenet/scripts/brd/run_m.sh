cd /root/SNN/code/Spikingformer-CML/imagenet
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29500' --use_env train.py -c config/brd/vit-m-brd.yml
