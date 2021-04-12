import torch
import torch.nn.functional as F

def quantize_bw(kernel):
    """
    binary quantization
    Return quantized weights of a layer.
    """
    delta = kernel.abs().mean()
    sign = kernel.sign().float()
    return sign*delta

def quantize_tnn(kernel):
    """
    ternary quantization
    Return quantized weights of a layer.
    """
    data = kernel.abs()
    delta = 0.7*data.mean()
    delta = min(delta, 100.0)
    index = data.ge(delta).float()
    sign = kernel.sign().float()
    scale = (data*index).mean()
    return scale*index*sign

def quantize_fbit(kernel):
    """
    four bit quantization
    """
    data = kernel.abs()
    delta = data.max()/15
    delta = min(delta, 10.0)
    sign = kernel.sign().float()
    q = 0.0*data

    for i in range(3,17,2):
        if i<15:
            index = data.gt((i-2)*delta).float()*data.le(i*delta).float()
        else:
            index = data.gt(13*delta).float()
        q += (i-1)/2*index
    
    scale = (data*q).sum()/(q*q).sum()
    return scale*q*sign


def identity(kernel):
    return kernel

def _accuracy(target, output, topk=(1,5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


quant_dict = {1: quantize_bw, 2: quantize_tnn, 4: quantize_fbit, 32: identity}


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# evaluate model with current weight
def _evaluate(model, loss, val_iterator):

    loss_value = AverageMeter()
    accuracy = AverageMeter()

    for j, (x_batch, y_batch) in enumerate(val_iterator):
        x_batch, y_batch = x_batch.to('cuda'), y_batch.to('cuda')

        logits = model(x_batch)

        # compute logloss
        batch_loss = loss(logits, y_batch)

        # compute accuracies
        pred = logits.data
        batch_accuracy = _accuracy(y_batch, pred, topk=(1,5))

        loss_value.update(batch_loss.item(), x_batch.size(0))
        accuracy.update(batch_accuracy[0], x_batch.size(0))
    return loss_value.avg, accuracy.avg.item()

def train(args, net, opt, loss, lr_scheduler, train_loader, valid_loader):
    proj = quant_dict[args.wbit]
    weights_quant = opt.param_groups[0]['params']
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    rec_weight = []
    for epoch in range(args.n_epochs):
        train_loss_value = AverageMeter()
        train_acc_value  = AverageMeter()
        net.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)
            if args.method != 'float':
                w_f = [k.data.clone() for k in weights_quant]
                for k in weights_quant:
                    k.data = proj(k.data)
            if args.method != 'float' and epoch >= args.n_epochs - 2:
                rec_layer = net.module.conv_1_3x3 if 'res' in args.arch else net.module.features.conv0
                rec_weight += rec_layer.weight.view(-1).tolist()
            logits = net(x)
            batch_loss = loss(logits, y)
            batch_accuracy = _accuracy(y, logits.data, topk=(1,5))
            train_loss_value.update(batch_loss.item(), x.size(0))
            train_acc_value.update(batch_accuracy[0], x.size(0))
            opt.zero_grad()
            batch_loss.backward()
            if args.method != 'float':
                for i, k in enumerate(weights_quant):
                    k.data = w_f[i].data
            opt.step()
        lr_scheduler.step()
        train_loss.append(train_loss_value.avg)
        train_acc.append(train_acc_value.avg.item())
        net.eval()
        if args.method == 'float':
            epoch_loss, epoch_acc = _evaluate(net, loss, valid_loader)
        else:
            w_f = [k.data.clone() for k in weights_quant]
            for k in weights_quant:
                k.data = proj(k.data)
            epoch_loss, epoch_acc = _evaluate(net, loss, valid_loader)
            valid_loss.append(epoch_loss)
            valid_acc.append(epoch_acc)
            for i, k in enumerate(weights_quant):
                k.data = w_f[i].data
        print('Epoch ' + str(epoch) + ':\t', epoch_acc, '\t, Best: ', max(epoch_acc, best_acc))
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            print('***Update Best Validation Accuracy***')
            torch.save(net.state_dict(), args.save_model)
    return train_loss, train_acc, valid_loss, valid_acc, rec_weight
