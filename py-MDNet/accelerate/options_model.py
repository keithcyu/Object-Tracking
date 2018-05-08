from collections import OrderedDict

opts_model = OrderedDict()

# layers to be decomposed
# recommend: conv2, conv2, fc4
opts_model['decomp_layers'] = ['conv2', 'conv3', 'fc4']
