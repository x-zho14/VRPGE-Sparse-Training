import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
from args import args as parser_args

class VRPGE(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()[0], 1, 1, 1))
        self.register_buffer('subnet', torch.zeros_like(self.scores))
        self.train_weights = False
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        if self.out_channels == 10 or self.out_channels == 100:
            self.prune = False
            self.subnet = torch.ones_like(self.scores)
        else:
            self.prune = True
        self.register_buffer("stored_mask_0", torch.zeros_like(self.scores))
        self.register_buffer("stored_mask_1", torch.zeros_like(self.scores))

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def forward(self, x):
        if self.prune:
            if not self.train_weights:
                self.subnet = StraightThroughBinomialSampleNoGrad.apply(self.scores)
                if parser_args.j == 0:
                    self.stored_mask_0.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                else:
                    self.stored_mask_1.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                w = self.weight * self.subnet
                x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                w = self.weight * self.subnet
                x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class StraightThroughBinomialSampleNoGrad(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return torch.zeros_like(grad_outputs)