import sys

from model_svd import *

sys.path.insert(0, '../modules')
from model import *

model_path = '../models/mdnet_vot-otb.pth'

MDNet_svd(model_path, 1)
