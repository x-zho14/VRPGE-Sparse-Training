# Efficient Neural Network Training via Forward and Backward Propagation Sparsification

Sparse training is a natural idea to accelerate the training speed of deep neural networks and save the memory usage, especially since large modern neural networks are significantly over-parameterized.  However, most of the existing methods cannot achieve this goal in practice because the chain rule based gradient (w.r.t. structure parameters) estimators adopted by previous methods require dense computation at least in the backward propagation step.  This paper solves this problem by proposing an efficient sparse training method with completely sparse forward and backward passes. We first formulate the training process as a continuous minimization problem under global sparsity constraint. We then separate the optimization process into two steps, corresponding to weight update and structure parameter update. For the former step, we use the conventional chain rule, which can be sparse via exploiting the sparse structure.  For the latter step, instead of using the chain rule based gradient estimators as in existing methods, we propose a variance reduced policy gradient estimator, which only requires two forward passes without backward propagation, thus achieving completely sparse training. We prove that the variance of our gradient estimator is bounded. Extensive experimental results on real-world datasets demonstrate that compared to previous methods, our algorithm is much more effective in accelerating the training process, up to an order of magnitude faster. 
## Requirements:

```
Pytorch 1.4
Python 3.7.7
CUDA Version 10.1
pyyaml 5.3.1
tensorboard 2.2.1
torchvision 0.5.0
tqdm 4.50.2
```
## Setup
1. Set up a virtualenv with python 3.7.7 with conda.
2. Install the required packages.
3. Create a data directory as a base for all datasets, e.g., ./data/ in the code directory/
## Demo
```bash
python main.py --config configs/config.yaml --multigpu 0 --score-init-constant 0.5 --prune-rate 0.5 --arch resnet32 --set CIFAR10 --lr 12e-3
```
## Implementation
1. The implementation of Variance-Reduced PGE can be found at utils/conv_type.py VRPGE and trainers/default.py calculateGrad.

##

## Cite
If you find this implementation is helpful to you, please cite:

```BibTeX
@article{zhou2021efficient,
  title={Efficient Neural Network Training via Forward and Backward Propagation Sparsification},
  author={Zhou, Xiao and Zhang, Weizhong and Chen, Zonghao and Diao, Shizhe and Zhang, Tong},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
