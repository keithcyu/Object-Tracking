import sys

import torch.nn as nn

from options import *

sys.path.insert(0, '../modules')
from model import *

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)

def svd_approx(model):
    
    print("model: %s" %type(model))
    print(type(model.layers.fc4))
    
    # this returns an iterator
    print(type(model.layers.fc5.parameters()))

    # 
    for name, param in model.layers.fc5.named_parameters():
        if name in ['1.weight']:
            print(name, param.size())
            print(type(param))
            U, S, V = param.svd()
            
            k = 200

            Uk = U[:, :k]
            Sk = S[:k]
            Vk = V[:, :k]

            US = torch.mm(U, torch.diag(S))
            USV = torch.mm(US, V.t())
            dist = torch.dist(param, USV)
            print(dist)
            # torch.dist(param, torch.mm(torch.mm(U, torch.diag(S)), V.t()))
            print(torch.dist(param, torch.mm(torch.mm(Uk, torch.diag(Sk)), Vk.t())))

    # put it back to model OMG how the fuck????
    # need to declare a whole new model.py... fuck...
    # I think I can just delete the ordered dict or layers and create a new one
    # what should the new layer be? 
    # it isn't linear anymore? or linear with 0 weight? 
    # no! the 0 weight will be trained
    # how do I stop the 0 weight from being trained?
    # can set linear layer.bias to false!!!!
    # forward function should stay the same for the new one
    # (forward function only applies the layers in sequence)

    # try to set model.layers

    # cannot delete, need to construct a new class
    # try to set model.layers
    # del model.layers['fc5']
    model.layers[4][0] = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, k, bias=False), nn.Linear(k, k, bias=False), nn.Linear(k, 512))
    
    # cannot assign try deleting the layer and create a new forward?
    print(model.layers[4])
    

# set parameters for new fc5
    for name, param in model.layers.fc5.named_parameters():
        print(name, param.size())

    # access fc4 params

    # svd
    
    # choose proper k (# of rank using) (guess one for now)
    
    # done?
# class MDNet_SVD(nn.Module)
    # def __init__(self, model_path=None, K=1):
        # super(MDNet, self).__init__()
        # self.K = K
        # self.layers = nn.Sequential

if __name__ == "__main__":

    # init model
    model = MDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    # call function
    svd_approx(model)



