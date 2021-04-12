import argparse
from utils import *
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets_th
from torch import optim

from quant import train, _evaluate, quant_dict
import numpy as np
from nets.vgg_type import QtUniVGG
from nets.resnet_type_cifar import QtUniRes
from nets.quant_uni_type import QuantReLU, bit_alpha

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--method', default='float', type=str, metavar='M',
                    help='training method')
parser.add_argument('--batch_size', default=128, type=int,
                    help='number of sample in one batch')
parser.add_argument('--ffun', default='quant', type=str, metavar='M',
                    help='forward function')
parser.add_argument('--bfun', default='inplt', type=str, metavar='M',
                    help='backward function')
parser.add_argument('--gd_alpha', action='store_true')
parser.add_argument('--gd_type', default='mean', type=str, metavar='M',
                    help='gradient method for alpha')
parser.add_argument('--arch', default='res20', type=str, metavar='M',
                    help='neural arch')
parser.add_argument('--num_classes', default=10, type=int,
                    help='number of classes')
parser.add_argument('--load_model', default='', type=str, metavar='M',
                    help='load model checkpoints')
parser.add_argument('--save_model', default='models/model.pkl', type=str, metavar='M',
                    help='load model checkpoints')
parser.add_argument('--n_epochs', default=200, type=int,
                    help='activation bit')
parser.add_argument('--abit', default=4, type=int, choices=[1, 2, 4, 32],
                    help='activation bit')
parser.add_argument('--wbit', default=1, type=int, choices=[1, 2, 4],
                    help='weight bit')
parser.add_argument('--ms1', default=80, type=int,
                    help='first milestone')
parser.add_argument('--ms2', default=140, type=int,
                    help='second milestone')
parser.add_argument('--lr', default=0.1, type=float, metavar='R',
                    help='initial learning rate')
parser.add_argument('--rate_factor', default=0.01, type=float, metavar='R',
                    help='lr_alpha / lr_weights')
parser.add_argument('--momentum', default=0.95, type=float, metavar='R',
                    help='momentum')
parser.add_argument('--decay_factor', default=1.0, type=float, metavar='R',
                    help='decay factor')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='R',
                    help='weight decay')
parser.add_argument('--device', default='cuda', type=str, metavar='M',
                    help='training device. cuda: GPU, cpu: cpu')
parser.add_argument('--rho', default=0.0, type=float, metavar='R',
                    help='blended parameter')

def main(args):
    factor = 20
    args.n_epochs *= factor
    args.batch_size *= factor
    args.ms1 *= factor
    args.ms2 *= factor
    # Prepare Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    trainset = datasets_th.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    validset = datasets_th.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=1000, shuffle=False, num_workers=args.workers)
    
    n_batches = len(train_loader)
    n_validation_batches = len(valid_loader)
    print("There are ", n_batches, " batches in the train set.")
    print("There are ", n_validation_batches, " batches in the val set.")

    # Init Model
    if args.method == 'float':
        ffun, bfun = 'relu', 'relu'
        gd_alpha = False
    else:
        ffun, bfun = args.ffun, args.bfun
        gd_alpha = True
    if 'vgg' in args.arch:
        QuantNet = QtUniVGG
    elif 'res' in args.arch:
        QuantNet = QtUniRes
    else:
        raise Exception('Only support ResNet and VGG')
    net = QuantNet(args.arch, num_classes = args.num_classes, bit = args.abit, ffun = ffun, bfun = bfun, rate_factor = args.rate_factor, gd_type = args.gd_type, gd_alpha=args.gd_alpha)
    net.apply(weight_init)
    net = net.to(args.device)
    if args.device == 'cuda':
        net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    # Prepare optimizer & loss function & scheduler
    # all fc layers and bias
    weights_float = [p for n, p in net.named_parameters() if 'weight' in n and 'conv' not in n] + [p for n, p in net.named_parameters() if 'bias' in n]
    # alpha
    weights_alpha = [p for n, p in net.named_parameters() if 'alpha' in n and 'gd_alpha' not in n]
    # all conv weights
    weights_quant = [p for n, p in net.named_parameters() if 'weight' in n and 'conv' in n]
    #
    #
    params = [
        {'params': weights_quant, 'weight_decay': args.weight_decay},
        {'params': weights_float, 'weight_decay': args.weight_decay},
        {'params': weights_alpha, 'weight_decay': args.weight_decay * args.decay_factor},
    ]
    opt = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[args.ms1, args.ms2], gamma=0.1)
    loss = nn.CrossEntropyLoss().to(args.device)

    # Load model
    if args.load_model:
        net.load_state_dict(torch.load(args.load_model))

    # Set trainable alpha
    if args.gd_alpha:
        net.apply(alpha_start_gd)
    # evaluate
    if args.evaluate:
        net.eval()
        method = 'float' if args.method == 'float' else 'binary' if args.wbit==1 else 'ternary'
        proj = quant_dict[args.wbit]
        print("Evaluation with " + method + ' weight')
        if method != 'float':
            for k in weights_quant:
                k.data = proj(k.data)
        _, acc = _evaluate(net, loss, valid_loader)
        print('Accuracy: ', acc)
        return
    # Train 
    train_loss, train_acc, valid_loss, valid_acc, rec_weight = train(args, net, opt, loss, lr_scheduler, train_loader, valid_loader)
    #plotloss(train_loss, args.arch + '.4a' +str(args.wbit) + 'b.train_loss')
    #plotacc(valid_acc, args.arch + '.4a' +str(args.wbit) + 'b.valid_acc')
    #name = args.arch[:3] + '.' + str(args.wbit) + 'b.dat'
    #if args.method != 'float':
    #    np.array(rec_weight).astype('float32').tofile(name)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
