from spiking_resnet import *
from sew_resnet import *
import utils
import torch
import os
from train import load_data
from spikingjelly.clock_driven import functional
from torchvision.models import *
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model import Spikingformer_base, Spikingformer_small
from timm.models import *
import timm
from model_mae import vit_base_patch16

model_dict = {"r18": resnet18, "r34": resnet34, "r50": resnet50, "vit-b": "vit_base_patch16_224", "vit-b-mae": vit_base_patch16, "vit-b-dino": "vit_base_patch16_224", "vit-s": "vit_small_patch16_224"}
model_sew_dict = {"r18": sew_resnet18, "r34": sew_resnet34, "r50": sew_resnet50}
model_spiking_dict = {"r18": spiking_resnet18, "r34": spiking_resnet34, "r50": spiking_resnet50}
model_spkformer = {"vit-b": Spikingformer_base, "vit-s": Spikingformer_small}
model_path = {
    "vit-s": "/home/zekai_xu/.cache/torch/hub/checkpoints/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "vit-b": "/home/zekai_xu/.cache/torch/hub/checkpoints/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "vit-b-dino": "/home/zekai_xu/.cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth",
    "vit-b-mae": "/home/zekai_xu/.cache/torch/hub/checkpoints/mae_finetuned_vit_base.pth"}


def load_model(model, path):
    olddict = torch.load(path)["model"]
    newdict = {}
    for k, v in olddict.items():
        newdict[k.replace('module.', '')] = v
    model.load_state_dict(newdict)

def load_dataloader(path):
    train_dir = os.path.join(path, 'train')
    val_dir = os.path.join(path, 'val')
    dataset_train, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, False, False)
    print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=16,
        sampler=train_sampler, num_workers=8, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=32,
        sampler=test_sampler, num_workers=8, pin_memory=True)
    return data_loader, data_loader_test

def get_metric(metric_logger, outputs, target, image):
    acc1, acc5 = utils.accuracy(outputs, target, topk=(1, 5))
    batch_size = image.shape[0]
    metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

def evaluation(model_name):
    model_spk = model_spkformer[model_name]().cuda()
    model_spk.load_state_dict(torch.load("../ckpt/checkpoint-306.pth.tar")["state_dict"], strict=True)
    name = "vit-b-mae"
    if name == "vit-b-mae":
        model = model_dict[name]().cuda()
        model.load_state_dict(torch.load(model_path[name])["model"], strict=True)
    else:
        model = timm.create_model(model_dict[name], pretrained=False,
                                  checkpoint_path=model_path[name]).cuda()
    data_loader, data_loader_test = load_dataloader("/nvme/ImageNet")

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger_spk = utils.MetricLogger(delimiter="  ")
    # metric_logger_sew_distillation = utils.MetricLogger(delimiter="  ")

    model.eval()
    model_spk.eval()

    te_tb_writer = SummaryWriter('logs_vit/te_b')
    # for i in range(10):
    #     x = np.random.random(1000)
    #     x = np.array(x, dtype='float')
    #     te_tb_writer.add_histogram('distribution centers', x + i, i)

    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader_test):
            image = image.cuda()
            target = target.cuda()
            outputs_spk, logits_spk = model_spk(image, True)
            outputs, logits = model(image, True)
            length = len(logits)
            functional.reset_net(model_spk)

            get_metric(metric_logger, outputs, target, image)
            get_metric(metric_logger_spk, outputs_spk, target, image)
            for idx, (l1, l2) in enumerate(zip(logits_spk, logits)):
                metric_logger_spk.meters['mse_layer{}'.format(str(idx))].update(torch.nn.MSELoss()(l1, l2).item(),
                                                                                n=image.shape[0])

                ls = [l1, l2]
                for j, l in enumerate(ls):
                    te_tb_writer.add_histogram('batch{}_layer_{}'.format(str(i + 1), str(idx + 1)), l.cpu().detach().numpy().flatten(), j)

            if i % 20 == 0:
                print(
                    "Vanilla {} acc: {:.2f}, SPK {} acc: {:.2f}, Spk logits gap: {:.4f}".format(model_name,
                        metric_logger.acc1.global_avg, model_name,
                        metric_logger_spk.acc1.global_avg,
                        metric_logger_spk.meters['mse_layer{}'.format(str(length - 1))].global_avg))
                print(
                    "Spk layer-1 gap: {:.4f}, Spk layer-2 gap: {:.4f}".format(
                        metric_logger_spk.meters['mse_layer{}'.format(str(0))].global_avg,
                        metric_logger_spk.meters['mse_layer{}'.format(str(1))].global_avg
                    ))

    metric_logger.synchronize_between_processes()
    metric_logger_spk.synchronize_between_processes()

    acc1, acc5 = metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    # logits_gap_sew, acc1_sew, acc5_sew = metric_logger_sew.logits_gap.global_avg, metric_logger_sew.acc1.global_avg, metric_logger_sew.acc5.global_avg
    # logits_gap_spiking, acc1_spiking, acc5_spiking = metric_logger_spiking.logits_gap.global_avg, metric_logger_spiking.acc1.global_avg, metric_logger_spiking.acc5.global_avg

    for j in range(length):
        print("Spk layer{} gap: {:.4f}".format(
            j,
            metric_logger_spk.meters['mse_layer{}'.format(str(j))].global_avg
        ))

if __name__ == '__main__':
    import torch.nn as nn
    import timm.models as m
    model_name = "vit-b"
    # model_sew = model_sew_dict[model_name](connect_f="ADD").cuda()
    # load_model(model_sew, '../ckpt/sew{}_checkpoint_319.pth'.format(model_name[1:]))
    # model = model_dict[model_name](pretrained=True).cuda()
    #
    # image = torch.randn((2,3,224,224)).cuda()
    # _, o1 = model_sew(image, True)
    # _, o2 = model(image, True)
    # print(len(o1), len(o2))
    # for s1, s2 in zip(o1, o2):
    #     print(s1.mean(), s1.min(), s1.max())
    #     print(s2.mean(), s2.min(), s2.max())

    # print(nn.ModuleList([model, model_sew]).parameters())
    # loss = torch.nn.MSELoss()(o1[-1], o2[-1])
    # loss.backward()
    evaluation(model_name)