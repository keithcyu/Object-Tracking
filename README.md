# Object Tracking Acceleration

## Introduction

Used [MDNet](https://github.com/HyeonseobNam/MDNet) to accelerate object tracking application. Acceleration method includes Tensor Decomposition of DL Parameters and Parameter Encoding

## Tensor Decomposition for Deep Learning Parameters

Training over-parameterized neural networks facilitates the convergence of the model to a local optima. However, the trained models are significantly redundant in the way they are represented. Once the model is trained and ready to deploy, this redundancy can be removed to improve inference efficiency. The computation and storage costs of CNNs are dominated by convolutional and fully-connected layers, respectively. We aim to utilize tensor decomposition techniques along with our automation tools to customize CNNs. Decomposing convolutional kernels and fully-connected weight matrices result in computation and storage optimizations, respectively.

## Parameter Encoding

This customization technique specifically targets the execution and dynamic updating of CNNs on reconfigurable devices such as FPGA. There are two challenges to be addressed in this domain. First, the memory requirement of CNNs often exceeds the on-chip storage capacity available in the FPGA, making it inevitable to store the parameters and the intermediate activation units in an off-chip memory; consequently, the achievable throughput becomes bounded by the memory access bandwidth. Second, The computation burden shall be reduced based on the specifications of the underlying hardware (the FPGA) and the required CNN accuracy.