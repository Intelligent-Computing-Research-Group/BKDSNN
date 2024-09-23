CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port='29501' train_distillation.py \
                                                           --cos_lr_T 320 \
                                                           --model sew_resnet50 -b 32 \
                                                           --model_teacher resnet50 \
                                                           --output-dir ./logs --tb --print-freq 64 \
                                                           --amp --cache-dataset --connect_f ADD --T 4 \
                                                           --lr 0.1 --epoch 320 --data-path /data/zkxu/DG/dataset/imagenet \
                                                           --distill_type mgd --teacher_channel 2048 --student_channel 2048