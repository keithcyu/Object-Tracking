import os
import sys
import scipy.io
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

import tensorly as tl
import tensorly

from decompositions import *
sys.path.insert(0, '../modules')
from model import *

def append_params(params, module, prefix):
    for child_name, child in module.named_children():
        for k,p in child._parameters.items():
            # print(prefix, child_name, child, k)
            if p is None: continue
            
            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            elif prefix == 'fc5':
                name = prefix + '_' + child_name + '_' + k
            else:
                name = prefix + '_' + k
            
            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))


class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq,pad,pad,pad,pad),2),
                            torch.cat((pad,x_sq,pad,pad,pad),2),
                            torch.cat((pad,pad,x_sq,pad,pad),2),
                            torch.cat((pad,pad,pad,x_sq,pad),2),
                            torch.cat((pad,pad,pad,pad,x_sq),2)),1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x


class MDNet_svd(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet_svd, self).__init__()
        self.K = K
        k = 1
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU())),
                ('fc4',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512 * 3 * 3, 512),
                                        nn.ReLU())),
                # fc5 should not have 3 linear layers
                # linear layers apply onto x for each layer
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLu()))]))
        """
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(k, 512, bias=False),
                                        nn.Linear(k, k, bias=False),
                                        nn.Linear(512, k, bias=True),
                                        nn.ReLU()))]))
        """
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5), 
                                                     nn.Linear(512, 2)) for _ in range(K)])
        
        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model_svd(model_path, k)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d'%(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False
 
    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params
    
    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6'):
        # forward model from in_layer to out_layer

        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                # TODO runtime debugging
                # print('running', name)
                # for name2, module2 in self.layers._modules[name].named_children():
                    # print(name, name2, type(module2))
                x = module(x)
                if name == 'conv3':
                    x = x.view(x.size(0),-1)
                if name == out_layer:
                    return x
        
        x = self.branches[k](x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x)
    
    def load_model_svd(self, model_path, k):
        
        # load the saved model
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        branches_layer = states['branches_layer']
        
        # load all layers
        self.layers.load_state_dict(shared_layers, strict=False)

        """
        # load layers except layer 5
        load_layers = {key: value for key, value in list(shared_layers.items()) if 'fc5' not in key}
        self.layers.load_state_dict(load_layers, strict=False)
        """

        # load braches
        self.branches.load_state_dict(branches_layer, strict=False)

        """
        # do svd on other layers
        fc5_weight = shared_layers['fc5.1.weight']
        fc5_bias = shared_layers['fc5.1.bias']
        U, S, V = fc5_weight.svd()

        # reduce rank
        Uk = U[:, :k]
        Sk = S[:k]
        Vk = V[:, :k]

        # write into layer
        self.layers[4][1].weight.data = Vk.t()
        self.layers[4][2].weight.data = torch.diag(Sk)
        self.layers[4][3].weight.data = Uk
        self.layers[4][3].bias.data = fc5_bias
        """

        ### TODO replace fully connected layer
        for i, key in enumerate(self.layers._modules.leys()):
            for i2, key2 in enumerate(self.layers._modules[key]._modules.keys()):
                layer = self.layers._modules[key]._modules[key2]
                if isinstance(layer, toch.nn.modules.Linear):
                    decomposed = svd_decomposition_fully_connected_layer(layer)
                    self.layers._modules[key]._modules[key2] = decomposed

        ### TODO replace all conv layers using tucker decomposition
        tl.set_backend('numpy')
        for i, key in enumerate(self.layers._modules.keys()):
            for i2, key2 in enumerate(self.layers._modules[key]._modules.keys()):
                print((i, key)) # success
                print((i2, key2)) # success

                if isinstance(self.layers._modules[key]._modules[key2], torch.nn.modules.conv.Conv2d):
                    conv_layer = self.layers._modules[key]._modules[key2]
                    decomposed = tucker_decomposition_conv_layer(conv_layer)
                    # TODO runtime debugging
                    # for i3, key3 in enumerate(decomposed._modules.keys()):
                        # print(i3, decomposed._modules[key3],)
                    self.layers._modules[key]._modules[key2] = decomposed
                    # print("decomposed", type(decomposed))

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)
    
    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]
        
        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i*4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:,0])

    

class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()
 
    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score, dim=0)[:,1]
        neg_loss = -F.log_softmax(neg_score, dim=0)[:,0]
        
        loss = pos_loss.sum() + neg_loss.sum()
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        
        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()
        
        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):
        
        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)
        
        return prec.data[0]
