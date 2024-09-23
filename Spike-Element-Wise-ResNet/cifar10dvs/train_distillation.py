import datetime
import os
import time
import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import sys
from torch.cuda import amp
import smodels
import argparse
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import cifar10_dvs
import math
from loss import *
import torch.nn as nn

_seed_ = 2020
import random
random.seed(2020)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)

from torch.utils.data import Dataset

class ConcatDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # Assuming both datasets have the same length
        assert len(dataset1) == len(dataset2), "Datasets must have the same length"

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        # Get samples and labels from both datasets
        sample1, label1 = self.dataset1[idx]
        sample2, label2 = self.dataset2[idx]

        assert label1 == label2

        return sample1, sample2, label1

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)

def main():
    parser = argparse.ArgumentParser(description='Classify DVS128 Gesture')
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-T_teacher', default=16, type=int, help='simulating time-steps for teacher')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', type=str, default='/nvme/cifar-10-dvs')
    parser.add_argument('-out_dir', type=str, help='root dir for saving logs and checkpoint', default='./logs')

    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')

    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam', default='SGD')
    parser.add_argument('-lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=32, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=64, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-model', type=str, default='SEWResNet')
    parser.add_argument('-cnf', type=str, default="ADD")
    parser.add_argument('-T_train', default=None, type=int)
    parser.add_argument('-dts_cache', type=str, default='./dts_cache')

    # for distillation
    parser.add_argument('--temp', default=2., type=float, metavar='TP',
                        help='temp')
    # parser.add_argument('--num_classes', default=1000, type=int, metavar='CLC',
    #                     help='number classes')
    parser.add_argument('--distill_weight', default=1., type=float, metavar='DW',
                        help='distill weight')
    parser.add_argument('--distill_type', default="mixed", type=str, metavar='DT',
                        help='distill type')
    parser.add_argument('--model_teacher', default='SEWResNet', help='SEWResNet')
    parser.add_argument('--teacher_channel', default=128, type=int, metavar='TC',
                        help='teacher_emb')
    parser.add_argument('--student_channel', default=128, type=int, metavar='SC',
                        help='student_emb')
    parser.add_argument('--alpha_mgd', default=2e-3, type=float, metavar='AM',
                        help='alpha_mgd')
    parser.add_argument('--lambda_mgd', default=0.15, type=float, metavar='LM',
                        help='lambda_mgd')
    parser.add_argument(
        "--distill",
        help="whether to use distillation",
        action="store_true",
    )
    parser.add_argument(
        "--use_clip",
        help="whether to use clip in brd",
        action="store_true",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs='+',
        help="whether to use clip in mgd"
    )

    args = parser.parse_args()
    print(args)


    net = smodels.__dict__[args.model](args.cnf, args.indices)
    print(net)
    net.to(args.device)

    net_teacher = smodels.__dict__[args.model](args.cnf,args.indices)
    net_teacher.load_state_dict(torch.load("ckpt/best.pth", map_location="cpu")["net"])
    net_teacher.to(args.device)
    net_teacher.eval()

    if args.distill:
        if args.distill_type == 'bkd' or args.distill_type == 'mixed':
            dist_loss = BRDLoss(teacher_channels=args.teacher_channel, student_channels=args.student_channel,
                             lambda_mgd=args.lambda_mgd, alpha_mgd=args.alpha_mgd, use_clip=args.use_clip).to(args.device)
        elif args.distill_type == "kdsnn":
            dist_loss = KDSNNLoss(teacher_channels=args.teacher_channel, student_channels=args.student_channel,
                               alpha_mgd=args.alpha_mgd).to(args.device)
        else:
            dist_loss = LaSNNLoss(teacher_channels=args.teacher_channel, student_channels=args.student_channel,
                               alpha_mgd=args.alpha_mgd, fnum=len(args.indices)).to(args.device)
        if args.distill_type != "wsld":
            dist_loss.to(args.device)

    optimizer = None
    if args.opt == 'SGD':
        if args.distill_type != "wsld":
            optimizer = torch.optim.SGD(nn.ModuleList([net, dist_loss]).parameters(), lr=args.lr, momentum=args.momentum)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'Adam':
        if args.distill_type != "wsld":
            optimizer = torch.optim.Adam(nn.ModuleList([net, dist_loss]).parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = None
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError(args.lr_scheduler)

    train_set_pth = os.path.join(args.dts_cache, f'train_set_{args.T}_{args.T_teacher}.pt')
    test_set_pth = os.path.join(args.dts_cache, f'test_set_{args.T}_{args.T_teacher}.pt')
    if os.path.exists(train_set_pth) and os.path.exists(test_set_pth):
        train_set = torch.load(train_set_pth)
        test_set = torch.load(test_set_pth)
    else:
        origin_set = cifar10_dvs.CIFAR10DVS(root=args.data_dir, data_type='frame', frames_number=args.T, split_by='number')

        train_set, test_set = split_to_train_test_set(0.9, origin_set, 10)
        if not os.path.exists(args.dts_cache):
            os.makedirs(args.dts_cache)
        torch.save(train_set, train_set_pth)
        torch.save(test_set, test_set_pth)

    train_data_loader = DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        num_workers=args.j,
        drop_last=True,
        pin_memory=True)

    test_data_loader = DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        num_workers=args.j,
        drop_last=False,
        pin_memory=True)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(args.out_dir, f'{args.model}_{args.cnf}_T_{args.T}_{args.T_teacher}_T_train_{args.T_train}_{args.opt}_lr_{args.lr}_')
    if args.lr_scheduler == 'CosALR':
        out_dir += f'CosALR_{args.T_max}'
    elif args.lr_scheduler == 'StepLR':
        out_dir += f'StepLR_{args.step_size}_{args.gamma}'
    else:
        raise NotImplementedError(args.lr_scheduler)

    if args.amp:
        out_dir += '_amp'


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    else:
        print(out_dir)
        # assert args.resume is not None

    pt_dir = out_dir + '_pt'
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)
        print(f'Mkdir {pt_dir}.')


    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        net_teacher.eval()
        train_loss = 0
        train_ce_loss = 0
        train_dist_loss = 0
        train_dist_feature_loss = 0
        train_dist_logits_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, frame_teacher, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.float().to(args.device)
            frame_teacher = frame_teacher.float().to(args.device)

            if args.T_train:
                sec_list = np.random.choice(frame.shape[1], args.T_train, replace=False)
                sec_list.sort()
                frame = frame[:, sec_list]

            label = label.to(args.device)
            if args.amp:
                with amp.autocast():
                    out_fr, out_feature_s = net(frame, True)
                    _, out_feature_t = net_teacher(frame_teacher, True)
                    if args.distill_type != "mixed":
                        if args.distill_type == "wsld":
                            loss_dist = args.distill_weight * get_logits_loss(out_feature_t[-1], out_feature_s[-1], F.one_hot(label, 10), args.temp, 10)
                        elif args.distill_type == "lasnn":
                            loss_dist = dist_loss(out_feature_t[:len(args.indices)], out_feature_s[:len(args.indices)])
                        else:
                            loss_dist = dist_loss(out_feature_t[len(args.indices)], out_feature_s[len(args.indices)])
                    else:
                        loss_dist_feat = dist_loss(out_feature_t[0], out_feature_s[0])
                        loss_dist_logits = args.distill_weight * get_logits_loss(out_feature_t[-1], out_feature_s[-1], F.one_hot(label, 10), args.temp, 10)
                        loss_dist = loss_dist_logits + loss_dist_feat
                    loss_ce = F.cross_entropy(out_fr, label)
                    loss = loss_ce + loss_dist
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr, out_feature_s = net(frame, True)
                _, out_feature_t = net_teacher(frame_teacher, True)
                if args.distill_type != "mixed":
                    if args.distill_type == "wsld":
                        loss_dist = args.distill_weight * get_logits_loss(out_feature_t[-1], out_feature_s[-1], F.one_hot(label, 10),
                                                                          args.temp, 10)
                    elif args.distill_type == "lasnn":
                        loss_dist = dist_loss(out_feature_t[:len(args.indices)], out_feature_s[:len(args.indices)])
                    else:
                        loss_dist = dist_loss(out_feature_t[len(args.indices)], out_feature_s[len(args.indices)])
                else:
                    loss_dist_feat = dist_loss(out_feature_t[0], out_feature_s[0])
                    loss_dist_logits = args.distill_weight * get_logits_loss(out_feature_t[-1], out_feature_s[-1],
                                                                             F.one_hot(label, 10), args.temp, 10)
                    loss_dist = loss_dist_logits + loss_dist_feat
                loss_ce = F.cross_entropy(out_fr, label)
                loss = loss_ce + loss_dist
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_ce_loss += loss_ce.item() * label.numel()
            train_dist_loss += loss_dist.item() * label.numel()
            if args.distill_type == "mixed":
                train_dist_logits_loss += loss_dist_logits.item() * label.numel()
                train_dist_feature_loss += loss_dist_feat.item() * label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)
        train_loss /= train_samples
        train_acc /= train_samples
        train_dist_loss /= train_samples
        train_ce_loss /= train_samples
        writer.add_scalar('train_ce_loss', train_ce_loss, epoch)
        writer.add_scalar('train_dist_loss', train_dist_loss, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        if args.distill_type == "mixed":
            train_dist_logits_loss /= train_samples
            train_dist_feature_loss /= train_samples
            writer.add_scalar('train_dist_logits_loss', train_dist_logits_loss, epoch)
            writer.add_scalar('train_dist_feature_loss', train_dist_feature_loss, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, _, label in test_data_loader:
                frame = frame.float().to(args.device)
                label = label.to(args.device)
                out_fr = net(frame)
                loss = F.cross_entropy(out_fr, label)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_latest.pth'))
        for item in sys.argv:
            print(item, end=' ')
        print('')
        print(args)
        print(out_dir)
        total_time = time.time() - start_time
        if args.distill_type=="mixed":
            print(
                f'epoch={epoch}, train_loss={train_loss}, train_dist_loss={train_dist_loss}, train_ce_loss={train_ce_loss}, train_dist_logits_loss={train_dist_logits_loss}, train_dist_feature_loss={train_dist_feature_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={total_time}, escape_time={(datetime.datetime.now() + datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')
        else:
            print(
                f'epoch={epoch}, train_loss={train_loss}, train_dist_loss={train_dist_loss}, train_ce_loss={train_ce_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={total_time}, escape_time={(datetime.datetime.now() + datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
