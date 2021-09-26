import argparse
import cv2
import os
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models

import torch.nn as nn
import time
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
#from models import *
import torch.nn.functional as F
import models
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from matplotlib import pyplot as plt
import math
import tensorflow as tf
from scipy import misc
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./test', type=str, metavar='PATHSAVE',
                    help='path to save the model (default: none)')

parser.add_argument('--dataset', type=str, default='mydataset',
                    help='training dataset (default: cifar10)')
parser.add_argument('--datapath', default='./RGB_Numpy_PAGENET64_FineTuned', type=str, metavar='PATH',
                    help='subject to train')
parser.add_argument('--subject', default='SubjectE', type=str, metavar='PATH',
                    help='subject to train')

parser.add_argument('--topkranking', type=int, default=1,
                    help='for top k ranking')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--normalization', type=int, default=0,
                    help='0 not normalization others normalization')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--toTensorform', type=int, default=0, help='0 for 0~1; 1 for -1~1; 2 for original type; 3 for normalization from original data; 4 for nomalization from toTensordata; 5 for (original-0.5)/0.5')

args = parser.parse_args()

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model.feature._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.avgpool = nn.AvgPool2d(2)
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        target_activations, output = self.feature_extractor(x)
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        #output = self.model.classifier(output) #############change############
        return target_activations, output     



class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)  #######
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (64, 64))
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam


print(torch.__version__)

args.cuda = not args.no_cuda and torch.cuda.is_available()
path_img_test = args.datapath+'/'+args.subject+'/x_img_test_D.npy'
path_label_test = args.datapath+'/'+args.subject+'/y_label_test.npy'
topk = args.topkranking

if args.topkranking<=0 or args.topkranking>=25:
    topk = 1


if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        
        checkpoint = torch.load(args.model)
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])

        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}\n"
                  .format(args.model, checkpoint['epoch'], best_prec1))

if args.cuda:
    model.cuda()


class MyDataset(Dataset):
    def __init__(self, path_img, path_label, transform=0):
        
        img_D = np.load(path_img)
        label_D = np.load(path_label)
        img_D = img_D[:,:,:,np.newaxis]

        #'0 for 0~1; 1 for -1~1; 2 for original type; 3 for normalization from original data; 4 for nomalization from toTensordata; 5 for (original-0.5)/0.5'
        if transform == 0 or transform ==1 or transform ==4:
            self.img_D = img_D.astype(np.uint8)
        else:
            self.img_D = img_D
        self.label_D = label_D.astype(float)
        
        # in this test: train and test mean or std calculate seperately (need to change in the future)
        if transform ==1 or transform==5:
            mean = [0.5]
            std = [0.5]
        elif transform == 3:
            mean = [np.mean(self.img_D[:,:,:,0])]
            std = [np.std(self.img_D[:,:,:,0])]
        elif transform == 4:
            mean = [np.mean(self.img_D[:,:,:,0])/(256)]
            std = [np.std(self.img_D[:,:,:,0])/(256*256)]
        else:
            mean = [0]
            std = [1]

        self.transform =  transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((mean[0],), (std[0],))])
        self.target_transform = None

    def __getitem__(self, index):

        img, label = self.img_D[index], self.label_D[index]
        img_o = img.copy()

        if self.transform is not None:
            img = self.transform(img)       
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img_o, img, label

    def __len__(self):
        return len(self.img_D)


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


def test(model,topk):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mydataset':
        test_loader = torch.utils.data.DataLoader(
            MyDataset(path_img_test, path_label_test, transform = args.toTensorform),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # Top-3
    
    top3 = AverageMeter()
    top5 = AverageMeter()
   

    model.eval()
    correct = 0
    pred_result = np.zeros(0)
    target_result = np.zeros(0)
    ierror = 0
    batchindex = 0
    numoforginaldata = np.zeros((1,25))

    grad_cam = GradCam(model = model, target_layer_names = ["10"], use_cuda=True)

    for ordata, data, target in test_loader:
        
        data_D = data.type(torch.FloatTensor)
        target = target.type(torch.LongTensor)
        target1 = target.cpu().numpy()

        if args.cuda:
            data_D, target = data_D.cuda() , target.cuda()

        undata_D, target = Variable(data_D, volatile=False), Variable(target)
        with torch.no_grad():
            testdata_D = Variable(data_D)
        output = model(testdata_D)  # <class 'torch.autograd.variable.Variable'> torch.Size([256, 25])
        
        # print(output[1])
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # array = [0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24];=
        arrayindex = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
        #print(torch.unsqueeze(data[0],0).shape)

        for inum in range(len(pred)):

            if int(target[inum])==14:# no_mind keep for other use
                # original data
                ordata_D = ordata
                Odata = ordata_D[inum][:,:,0].numpy()  
                Ddata = np.zeros((Odata.shape[0],Odata.shape[1]),dtype = np.uint8)
                
                for ix in range(Odata.shape[0]):
                    for iy in range(Odata.shape[1]):
                        for iz in range(3):
                            Ddata[ix,iy]=int(Odata[ix,iy])
               
                # generate original data for original image
                #im = Image.fromarray(RGBdata,'RGB')
                if int(pred[inum])==int(target[inum]):

                    savestream = (args.save+"/"+args.subject+"/pred_"+arrayindex[int(target[inum])]+"_"+arrayindex[int(pred[inum])])
		    # print(savestream)
                        
                    if not os.path.exists(savestream):
                        os.makedirs(savestream)
                        
                    # save original picture
                    #im.save(savestream+"/"+str(256*batchindex+inum)+".png")
                        
                    ## 需要修改
                    # save cam
                    input_D = torch.unsqueeze(undata_D[inum],0).float()
                    target_index = int(target[inum])
                    heat_map = grad_cam(input_D, target_index)
                    plt.imshow(Ddata,cmap='gray')
                    plt.imshow(heat_map,cmap='jet',alpha=0.4)
                    plt.title("("+arrayindex[int(target[inum])]+","+arrayindex[int(pred[inum])]+","+arrayindex[int(target[inum])]+")")
                    plt.savefig(savestream+"/"+str(64*batchindex+inum)+".png")

        batchindex = batchindex+1
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # view_as : distribute new tensor for target.data which have the same tensor size as "pred"

        # Top-3 Top-5
        
        prec3, prec5 = accuracy(output, target, topk=(3, 5))
        n = data_D.size(0)
        top3.update(prec3.item(), n)
        top5.update(prec5.item(), n)
  
    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
          correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

    print("top1: ",100. * correct / len(test_loader.dataset), "top3: ",top3.avg, "top5: ", top5.avg) 

test(model,topk)

