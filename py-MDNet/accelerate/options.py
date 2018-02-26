from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = False

opts['init_model_path'] = '../models/imagenet-vgg-m.mat' # pretrain
opts['model_path'] = '../models/mdnet_vot-otb.pth'

opts['img_size'] = 107
opts['padding'] = 16

opts['batch_frames'] = 8 # pretrain
opts['batch_pos'] = 32
opts['batch_neg'] = 96
opts['batch_neg_cand'] = 1024
opts['batch_test'] = 256

opts['n_samples'] = 256
opts['trans_f'] = 0.6
opts['scale_f'] = 1.05
opts['trans_f_expand'] = 1.5

opts['n_bbreg'] = 1000
opts['overlap_bbreg'] = [0.6, 1]
opts['scale_bbreg'] = [1, 2]

opts['lr_init'] = 0.0001
opts['maxiter_init'] = 30
opts['n_pos_init'] = 500
opts['n_neg_init'] = 5000
opts['overlap_pos_init'] = [0.7, 1]
opts['overlap_neg_init'] = [0, 0.5]

opts['lr_update'] = 0.0002
opts['maxiter_update'] = 15
opts['n_pos_update'] = 50
opts['n_neg_update'] = 200
opts['overlap_pos_update'] = [0.7, 1]
opts['overlap_neg_update'] = [0, 0.3]
opts['overlap_pos'] = [0.7, 1] # pretrain
opts['overlap_neg'] = [0, 0.5] # pretrain

opts['success_thr'] = 0
opts['n_frames_short'] = 20
opts['n_frames_long'] = 100
opts['long_interval'] = 10

opts['lr'] = 0.0001 # pretrain
opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10
# opts['lr_mult'] = {'fc6':10} # commented out by pretrain
opts['lr_mult'] = {'fc':10}
# opts['ft_layers'] = ['fc'] # commented out by pretrain
opts['ft_layers'] = ['conv','fc']
opts['n_cycles'] = 50 # pretrain
