from model import *
from loss import *
import torch
from torch.optim import AdamW
from timm.models import *
import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP

if __name__ == '__main__':
    import timm
    model = create_model(
        'Spikingformer',
        drop_rate=0.,
        drop_path_rate=0.2,
        drop_block_rate=None,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=384, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        depths=8, sr_ratios=1,
        T=4,
    ).cuda()

    path = "/home/zekai_xu/.cache/torch/hub/checkpoints/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    model_teacher = timm.create_model('vit_small_patch16_224', pretrained=False, checkpoint_path=path).cuda()
    print(model_teacher.head.weight.data)

    # model = NativeDDP(model, device_ids=[1], find_unused_parameters=True)
    # model_teacher = NativeDDP(model_teacher, device_ids=[1], find_unused_parameters=True)
    #
    # model_teacher.eval()
    # model.train()
    #
    # input = torch.randn(2,3,224,224).cuda()
    # label = torch.randint(0, 1000, size=(2,))
    # label = torch.nn.functional.one_hot(label, num_classes=1000).float().cuda()
    #
    # _, o1 = model(input, True)
    # _, o2 = model_teacher(input, True)
    # # loss = get_logits_loss(o1[-1], o2[-1], label, 2.)
    # loss_fn = BRDLoss(384, 384, 7e-3, 0.15, False).cuda()
    # optimizer = AdamW(torch.nn.ModuleList([model, loss_fn]).parameters())
    # loss = loss_fn(o1[0], o2[0])
    # print(loss)
    #
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # print(model)