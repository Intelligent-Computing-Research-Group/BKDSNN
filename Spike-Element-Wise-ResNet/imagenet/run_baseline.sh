CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port='29501' train.py \
                                                           --cos_lr_T 320 \
                                                           --model sew_resnet18 -b 128 \
                                                           --output-dir ./logs_baseline --tb --print-freq 64 \
                                                           --amp --cache-dataset --connect_f ADD --T 4 \
                                                           --lr 0.1 --epoch 320 --data-path /nvme/ImageNet