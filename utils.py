import os
import sys
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from einops import rearrange

def mkdirs(*args):
    for arg in args:
        if not os.path.exists(arg):
            os.makedirs(arg)

class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    """计算指定 k 值的前 k 个预测的准确率"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred[0]

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    loss_all = AverageMeter()
    acc_all = AverageMeter()
    tar_list, pre_list = [], []

    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(output, target)
        loss_all.update(loss.item(), data.size(0))
        acc_all.update(prec1[0].item(), data.size(0))
        
        tar_list.append(t.cpu().numpy())
        pre_list.append(p.cpu().numpy())

    return acc_all.avg, loss_all.avg, np.concatenate(tar_list), np.concatenate(pre_list)

def valid_epoch(model, valid_loader, criterion):
    model.eval()
    tar_list, pre_list = [], []

    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            _, _, p = accuracy(output, target)
            
            tar_list.append(target.cpu().numpy())
            pre_list.append(p.cpu().numpy())

    return np.concatenate(tar_list), np.concatenate(pre_list)

class Logger(object):
    """标准输出与文件同步记录器"""
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode='w'):
        self.file = open(file, mode)

    def write(self, message):
        self.terminal.write(message + '\n')
        if self.file:
            self.file.write(message + '\n')

    def flush(self): pass

    def close(self):
        if self.file: self.file.close()

def train_patch(data, label, patchsize, pad_width):
    """
    针对高光谱图像进行 Patch 提取与预处理
    返回维度: (Batch, Channel, Spectral, Height, Width)
    """
    m, n, l = data.shape
    # 归一化
    for i in range(l):
        band = data[:, :, i]
        data[:, :, i] = (band - band.min()) / (band.max() - band.min())

    # 边界填充
    data_pad = np.pad(data, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='symmetric')

    # 提取有标签像素的 Patch
    ind1, ind2 = np.where(label > 0)
    train_num = len(ind1)
    patches = np.empty((train_num, l, patchsize, patchsize), dtype='float32')
    labels = label[ind1, ind2]

    for i in range(train_num):
        r, c = ind1[i] + pad_width, ind2[i] + pad_width
        patch = data_pad[r-pad_width:r+pad_width+1, c-pad_width:c+pad_width+1, :]
        patches[i] = np.transpose(patch, (2, 0, 1))

    # 转换为 Tensor 并调整维度 (B, C, S, H, W)
    patches = torch.from_numpy(patches).unsqueeze(1) # Add dummy channel dim
    patches = rearrange(patches, 'b c s h w -> b c s h w') # 保持 5D 结构
    labels = torch.from_numpy(labels).long() - 1 

    return patches, labels

class ResidualBlock(nn.Module):
    def __init__(self, submodule):
        super().__init__()
        self.sub = submodule

    def forward(self, x):
        return x + self.sub(x)

def record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, testing_time_ae, path):
    f = open(path, 'a')

    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n' + '\n'
    f.write(sentence5)
    sentence7 = 'average Testing time is: ' + str(np.mean(testing_time_ae)) + '\n' + '\n'
    f.write(sentence7)

    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    sentence8 = "Mean of all elements in confusion matrix: " + str(element_mean) + '\n'
    f.write(sentence8)
    sentence9 = "Standard deviation of all elements in confusion matrix: " + str(element_std) + '\n'
    f.write(sentence9)

    f.close()
