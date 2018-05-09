# MDNet acceleration using Tensor Decomposition

## Prerequisites
- python 3.#
- [Tensorly](https://github.com/tensorly/tensorly)


## Usage

### Pretraining and Tracking
- please refer to the README for py-MDNet at the parent directory

### Decomposition
- edit **options.py** for training hyper parameters
- edit **options_model.py** for layers to decompose
'''bash
python3 train_mdnet_svd.py
'''

# References
- py-MDNet package: https://github.com/HyeonseobNam/py-MDNet
- Tensor Decomposition package: https://github.com/jacobgil/pytorch-tensor-decompositions
- VBMF package: https://github.com/CasvandenBogaard/VBMF
- CP Decomposition for convolutional layers is described here: https://arxiv.org/abs/1412.6553
- Tucker Decomposition for convolutional layers is described here: https://arxiv.org/abs/1511.06530
- VBMF for rank selection is described here: http://www.jmlr.org/papers/volume14/nakajima13a/nakajima13a.pdf
- Tensorly: https://github.com/tensorly/tensorly
