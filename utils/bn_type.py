import torch.nn as nn

LearnedBatchNorm = nn.BatchNorm2d
Identity = nn.Identity
import torch.nn.functional as F

class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

# class BatchNorm2dSpeedUp(nn.BatchNorm2d):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
#                  track_running_stats=False):
#         super(BatchNorm2dSpeedUp, self).__init__(
#             num_features, eps, momentum, affine, track_running_stats)
#     def forward(self, input, mask):
#         self._check_input_dim(input)
#
#         # exponential_average_factor is set to self.momentum
#         # (when it is available) only so that if gets updated
#         # in ONNX graph when this node is exported to ONNX.
#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum
#
#         if self.training and self.track_running_stats:
#             # TODO: if statement only here to tell the jit to skip emitting this when it is None
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked = self.num_batches_tracked + 1
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum
#
#         return F.batch_norm(
#             input, self.running_mean, self.running_var, self.weight, self.bias,
#             self.training or not self.track_running_stats,
#             exponential_average_factor, self.eps), mask
#
# class ReluSpeedUp(nn.ReLU):
#     def __init__(self, dim):
#         super(ReluSpeedUp, self).__init__(inplace=True)
#     def def forward(self, input, mask):
