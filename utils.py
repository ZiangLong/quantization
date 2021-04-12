import torch.nn as nn
import math
from nets.quant_uni_type import QuantReLU, bit_alpha
import matplotlib.pyplot as plt
import numpy as np

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def alpha_start_gd(m):
    if isinstance(m, QuantReLU):
        m.gd_alpha.fill_(1)

def alpha_init(m):
    if isinstance(m, QuantReLU):
        m.alpha.fill_(bit_alpha[args.abit])

def plotacc(x, name, m = 0):
    fig, ax = plt.subplots()
    ax.plot(x, label=str(100.0*max(x[m:]))+'%')
    legend = ax.legend(loc='lower right', shadow=True)
    savename = name+'.png'
    plt.savefig(savename)
    savename = name+'.txt'
    np.savetxt(savename, x ,fmt='%.6f')
    print(max(x[m:]))

def plotloss(x, name, m = 0):
    fig, ax = plt.subplots()
    epoch = np.linspace(1 ,len(x), len(x))
    ax.plot(epoch,x,label=str(min(x[m:])))
    legend = ax.legend(loc='upper right', shadow=True)
    savename = name+'.png'
    plt.savefig(savename)
    savename = name+'.txt'
    np.savetxt(savename,x,fmt='%.6f')
    print(min(x[m:]))