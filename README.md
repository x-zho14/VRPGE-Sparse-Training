# Efficient Neural Network Training via Forward and Backward Propagation Sparsification

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
