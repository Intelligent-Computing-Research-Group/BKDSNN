CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_distillation.py \
                                                           --cos_lr_T 320 \
                                                           --model sew_resnet18 -b 128 \
                                                           --model_teacher resnet34 \
                                                           --output-dir ./logs --tb --print-freq 64 \
                                                           --amp --cache-dataset --connect_f ADD --T 4 \
                                                           --lr 0.1 --epoch 320 --data-path /nvme/ImageNet \
                                                           --distill_type mgd --teacher_channel 512 --student_channel 512