import sys
import numpy as np

import tensorflow as tf
from keras import backend as K

sys.path.append('../pretrain')
from options import *

def BinaryLoss(y_true, y_pred):

    # ignore y_true

    pos_score = y_pred[ : opts['batch_pos'] , :]
    neg_score = y_pred[ opts['batch_pos'] : , :]

    pos_loss = -tf.nn.log_softmax(pos_score)[:,1]
    neg_loss = -tf.nn.log_softmax(neg_score)[:,0]
    
    # loss = pos_loss.sum() + neg_loss.sum()
    loss = K.sum(pos_loss) + K.sum(neg_loss)
    return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        
        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()
        
        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


'''
class Precision_class():
    def __call__(self, pos_score, neg_score):
        
    #scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
    scores = y_pred[:,1]
    #topk = torch.topk(scores, pos_score.size(0))[1]
    topk = tf.top_k(scores, opts['batch_pos'])[1]
    prec = (topk < opts['batch_pos']).float().sum() / (opts['batch_pos']+1e-8)

    return prec.data[0]
'''

def Precision(y_true, y_pred):
        
    #scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
    scores = y_pred[:,1]
    #topk = torch.topk(scores, pos_score.size(0))[1]
    topk = tf.nn.top_k(scores, opts['batch_pos'])[1]
    # prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)
    prec = K.sum(tf.to_float(topk < opts['batch_pos'])) / tf.to_float(opts['batch_pos']+1e-8)
    
    # return prec.data[0]
    return prec
