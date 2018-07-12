# MDNet acceleration using Tensor Decomposition

## Prerequisites
- python 3.#
- [Tensorly](https://github.com/tensorly/tensorly)


## Usage

### Pretraining
- please use py-MDNet to pretrain a model and copy the model into Decomposition directory
```bash
cp py-MDNet/models/mdnet_vot-otb.pth decomposition/models/mdnet_vot-otb.pth
```

### Decomposition
```bash
cd decomposition/
```
- edit **options.py** for training hyper parameters
- edit **options_model.py** for layers to decompose
```bash
python3 train_mdnet_svd.py
```

### Tracking
```bash
 cd tracking
 python run_tracker.py -s DragonBaby [-d (display fig)] [-f (save fig)]
```
 - You can provide a sequence configuration in two ways (see tracking/gen_config.py):
   - ```python run_tracker.py -s [seq name]```
   - ```python run_tracker.py -j [json path]```

# References
- py-MDNet package: https://github.com/HyeonseobNam/py-MDNet
- Tensor Decomposition package: https://github.com/jacobgil/pytorch-tensor-decompositions
- VBMF package: https://github.com/CasvandenBogaard/VBMF
- CP Decomposition for convolutional layers is described here: https://arxiv.org/abs/1412.6553
- Tucker Decomposition for convolutional layers is described here: https://arxiv.org/abs/1511.06530
- VBMF for rank selection is described here: http://www.jmlr.org/papers/volume14/nakajima13a/nakajima13a.pdf
- Tensorly: https://github.com/tensorly/tensorly
