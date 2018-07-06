# MDNet acceleration using Parameter Encoding (ReBNet)

## Prerequisites
- Keras
- Tensorflow

## Usage

### Pretraining
 - Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "models/imagenet-vgg-m.mat"
 - Download [VOT](http://www.votchallenge.net/) datasets into "dataset/vot201x"
 - edit **binarization_utils.py** Line 31 for **residual levels**
``` bash
 cd pretrain
 python prepro_data.py
 python train_mdnet.py
```

# References
- ReBNet API: https://github.com/mohaghasemzadeh/ReBNet
