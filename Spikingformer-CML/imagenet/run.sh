CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port='29501' --use_env train.py -c vit-s-brd.yml