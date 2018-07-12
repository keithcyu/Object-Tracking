import os
import sys
import pickle
import time

import numpy as np
# import torch
# import torch.optim as optim
# from torch.autograd import Variable

from keras import optimizers

from data_prov import *
from options import *


sys.path.append('../modules')
from model import *
from LR_SGD import *

# img_home = '../dataset/'
data_path = 'data/vot-otb.pkl'
img_home = os.path.expanduser('~/Object_Tracking/py-MDNet/dataset/')

'''
def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay'], clipnorm=opts['grad_clip']):
    
    # 
    params = model.get_learnable_params()
    # param_list = []
    lr_mult_dict = {}
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                lr = lr_base * m
        #param_list.append({'params': [p], 'lr':lr})
        lr_mult_dict[p] = lr
    #optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    # 
    # optimizer = optimizers.SGD(lr=lr_base, momentum=momentum, decay = w_decay, clipnorm=clipnorm)
    optimizer = LR_SGD(lr=lr_base, momentum = momentum, decay = w_decay, multipliers = lr_mult, clipnorm = clipnorm)
    return optimizer
'''

def train_mdnet():
    
    ## Init dataset ##
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)

    K = len(data)
    dataset = [None]*K
    for k, (seqname, seq) in enumerate(data.iteritems()):
        img_list = seq['images']
        gt = seq['gt']
        img_dir = os.path.join(img_home, seqname)
        dataset[k] = RegionDataset(img_dir, img_list, gt, opts)

    ## Init model ##
    print opts['init_model_path']
    model = get_model(opts['init_model_path'], K)
    # model = MDNet(opts['init_model_path'], K)
    # if opts['use_gpu']:
        # model = model.cuda()
    #  model.set_learnable_params(opts['ft_layers'])
        
    ## Init criterion and optimizer ##
    #criterion = BinaryLoss()
    #evaluator = Precision_class()
    optimizer = set_optimizer(model, opts['lr']) # only base learning rate is used

    model.compile(loss=BinaryLoss, optimizer=optimizer, metrics=[Precision])

    best_prec = 0.
    for i in range(opts['n_cycles']):
        print "==== Start Cycle %d ====" % (i)
        k_list = np.random.permutation(K)
        prec = np.zeros(K)
        for j,k in enumerate(k_list):

            file = open('k.txt','w') 
            file.write(str(k))
            file.close()

            tic = time.time()
            pos_regions, neg_regions = dataset[k].next()
            
            # pos_regions = Variable(pos_regions)
            # neg_regions = Variable(neg_regions)
            all_regions = np.concatenate((pos_regions, neg_regions), axis=0)
            all_regions = all_regions.transpose([0, 2, 3, 1]) # put channel to last dimension
        
            if opts['use_gpu']:
                # pos_regions = pos_regions.cuda()
                # neg_regions = neg_regions.cuda()
                # all_regions = all_regions.cuda()
                pass
        
            # pos_score = model(pos_regions, k)
            # neg_score = model(neg_regions, k)
            #model.compile(loss=BinaryLoss, optimizer=optimizer, metrics=[Precision])

            # Y_train = np.empty([opts['batch_pos']+opts['batch_neg'], 2]) # figure out output size? 
            Y_train = np.empty([all_regions.shape[0], 2])
            history =  model.fit(all_regions,
                        Y_train,
                        # batch_size=min(opts['batch_pos'], opts['batch_neg']),
                        batch_size=(opts['batch_pos'] + opts['batch_neg']),
                        epochs=1, verbose=0, shuffle=False)

            # print history.history.keys()
            # print history.history['loss'][0]
            # print history.history['Precision'][0]

            # loss = criterion(pos_score, neg_score)
            # model.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip']) # included in optimizer
            # optimizer.step()
            
            #prec[k] = evaluator(pos_score, neg_score) # don't have access to score, cannot calculate
            loss = history.history['loss'][0]
            prec[k] = history.history['Precision'][0]

            toc = time.time()-tic
            # print "Cycle %2d, K %2d (%2d), Loss %.3f, Prec %.3f, Time %.3f" % \
                    # (i, j, k, loss.data[0], prec[k], toc)
            print "Cycle %2d, K %2d (%2d), Loss %.3f, Prec %.3f, Time %.3f" % \
                    (i, j, k, loss, prec[k], toc)
        
        cur_prec = prec.mean()
        print "Mean Precision: %.3f" % (cur_prec)
        if cur_prec > best_prec:
            best_prec = cur_prec
            
            # if opts['use_gpu']:
                # model = model.cpu()
            # save both shared and branches
            # states = {'shared_layers': model.layers.state_dict()}
            # states['branches_layer'] = model.branches.state_dict()
            print "Save model to %s" % opts['model_path']
            
            # torch.save(states, opts['model_path'])
            # if opts['use_gpu']:
                # model = model.cuda()
        


if __name__ == "__main__":
    train_mdnet()

