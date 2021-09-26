import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import time
import models
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
from models import *
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from matplotlib import pyplot as plt

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='mydataset',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--subject', default='SubjectA/', type=str, metavar='PATH',
                    help='subject to train')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--datapath', default='./RGB_Numpy_PAGENET64_FineTuned', type=str, metavar='PATH',
                    help='subject to train')

print(torch.__version__)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

subject = args.subject

path_img_test = args.datapath+'/' + subject + '/x_img_test_D.npy'
path_label_test = args.datapath+'/' + subject + '/y_label_test.npy'

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)  # cfg=checkpoint['cfg']
        # print(checkpoint['cfg'])
        # print(cfg)

        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}\n"
              .format(args.model, checkpoint['epoch'], best_prec1))

        #print(checkpoint['state_dict'])

    else:
        print("=> no checkpoint found at '{}'\n".format(args.resume))

if args.cuda:
    model.cuda()


class MyDataset(Dataset):
    def __init__(self, path_img, path_label, transform=None, target_transform=None):

        img_D = np.load(path_img)
        label_D = np.load(path_label)
        img_D = img_D[:,:,:,np.newaxis]

        self.img_D = img_D.astype(np.uint8)
        self.label_D = label_D.astype(float)############change############

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        img, label = self.img_D[index], self.label_D[index]

        # print(img.shape, type(img)) # (64, 64, 4) <class 'numpy.ndarray'>
        # print(label, type(label)) # 2.0 <class 'numpy.float64'>

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        # print(img.shape, type(img)) # torch.Size([4, 64, 64]) <class 'torch.FloatTensor'>
        # print(label, type(label)) # 2.0 <class 'numpy.float64'>

        return img, label

    def __len__(self):
        return len(self.img_D)

# Top-3
'''
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n = 1):
        self.val = val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
'''


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mydataset':
        test_loader = torch.utils.data.DataLoader(
            MyDataset(path_img_test, path_label_test,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
        # print(len(test_loader)) # 50

    # Top-3
    '''
    top1 = AverageMeter()
    top3 = AverageMeter()
    '''

    model.eval()
    correct = 0
    pred_result = np.zeros(0)
    target_result = np.zeros(0)

    for data, target in test_loader:
        # print(type(data), type(target))
        target = target.type(torch.LongTensor)
        target1 = target.cpu().numpy()

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        output = model(data)  # <class 'torch.autograd.variable.Variable'> torch.Size([256, 25])

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # view_as : distribute new tensor for target.data which have the same tensor size as "pred"

        # Top-3
        '''
        prec1, prec3 = accuracy(output, target, topk=(1, 3))
        n = data.size(0)
        top1.update(prec1.data[0], n)
        top3.update(prec3.data[0], n)
        '''


        pred = torch.squeeze(pred)
        pred = pred.cpu().numpy()
        pred_result = np.concatenate((pred_result, pred))  # <class 'numpy.ndarray'> (12547,)
        pred_result = pred_result.astype(int)

        target_result = np.concatenate((target_result, target1))  # <class 'numpy.ndarray'> (12547,)
        target_result = target_result.astype(int)

    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
          correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

    # print(top1.avg, top3.avg)
    return target_result, pred_result


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'


    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)  # <class 'numpy.ndarray'> (24, 24)

    # print(cm)

    if normalize:
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_xlabel('Predicted label', fontsize=16)
    ax.set_ylabel('True label', fontsize=16)

    # ax.xaxis.label.set_size(14)
    # ax.yaxis.label.set_size(20)

    # Rotate the tick labels and set their alignment.
    """
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    """

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    fmt1 = '.0f'

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt if cm[i, j]>0.01 else fmt1),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize= 6 if normalize else 6)

    plt.tick_params(labelsize=14)
    fig.tight_layout()
    plt.savefig('confusion_matrix.png')
    return ax


np.set_printoptions(precision=2)

y_test, y_pred = test(model)
np.savetxt("prediction.txt",y_pred,fmt="%1f")
np.savetxt("label.txt",y_test,fmt="%1f")

# array = [a b c d e f g h i k  l  m  n  o  p  q  r  s  t  u  v  w  x  y ]
# array = [0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
class_names = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'])
# Plot normalized confusion matrix

plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=False,
                      title=None)  # Confusion matrix, without normalization
                                                            # Normalized confusion matrix
plt.show()


