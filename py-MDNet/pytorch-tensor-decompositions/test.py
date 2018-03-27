# try calling decomposition.tucker_decomposition_conv_layer
# takes in a layer
# returns nn.Sequential(*new_layer)

import torch
import torch.nn as nn
from decompositions import *

# load svd decomposed model
model = torch.load("../models/mdnet_svd_vot-otb.pth")


# N = len(model.features._modules.keys())
# print(N)

for m in model.layers:
    print(type(m))

for i, key in enumerate(model.features._modules.keys()):

    # if i >= N - 2:
        # break
    if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
        conv_layer = model.features._modules[key]
        decomposed = tucker_decomposition_conv_layer(conv_layer)

        model.features._modules[key] = decomposed

torch.save(model, 'decomposed_model')
