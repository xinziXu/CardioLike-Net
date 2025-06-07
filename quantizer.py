"""
Author@Xinzi Xu

This code is released for academic and research purposes only, as part of the publication:
"CardioLike-Net: An Edge-end Inter-patient Arrhythmia Classifier with Quantization-aware-training for Wearable ECG Applications"

Permission is granted to use, copy, and modify the code **for non-commercial research purposes**, provided that proper citation is given to the above paper.

Any commercial use of this code or any derivative work is strictly prohibited without explicit written permission from the author.

If you find this work useful in your research, please consider citing the following paper: XXX
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function
import copy
from config import get_config
config, unparsed = get_config()


class ObserverBase(nn.Module):
    def __init__(self):
        super(ObserverBase, self).__init__()

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        min_val = torch.min(input)
        max_val = torch.max(input)
        self.update_range(min_val, max_val)


class MinMaxObserver(ObserverBase):
    def __init__(self):
        super(MinMaxObserver, self).__init__()
        self.num_flag = 0
        self.use_gpu = config.use_gpu
        if self.use_gpu:
            self.device = config.device
        else:
            self.device = 'cpu'
        self.min_val = torch.zeros((1), dtype=torch.float32, device = self.device)
        self.max_val = torch.zeros((1), dtype=torch.float32, device = self.device)

    def update_range(self, min_val_cur, max_val_cur):
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class MovingAverageMinMaxObserver(ObserverBase):
    def __init__(self, momentum=0.1):
        super(MovingAverageMinMaxObserver, self).__init__()
        self.momentum = momentum
        self.num_flag = 0
        self.use_gpu = config.use_gpu
        if self.use_gpu:
            self.device = config.device
        else:
            self.device = 'cpu'
        self.min_val = torch.zeros((1), dtype=torch.float32, device = self.device)
        self.max_val  = torch.zeros((1), dtype=torch.float32, device = self.device)

    def update_range(self, min_val_cur, max_val_cur):
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class Round(torch.autograd.Function):
    @staticmethod
    def forward(self, input, observer_min_val, observer_max_val, q_type):
        # 对称
        if q_type == 0:
            max_val = torch.max(
                torch.abs(observer_min_val), torch.abs(observer_max_val)
            )
            min_val = -max_val
        # 非对称
        else:
            max_val = observer_max_val
            min_val = observer_min_val
        self.save_for_backward(input, min_val, max_val)
        sign = torch.sign(input)
        # output = sign * torch.floor(torch.abs(input) + 0.5)
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, min_val, max_val = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.gt(max_val)] = 0
        grad_input[input.lt(min_val)] = 0
        return grad_input, None, None, None

import math
class Quantizer(nn.Module):
    def __init__(self, bits, observer, reduce_range = False, bit_scale = 15):
        super(Quantizer, self).__init__()
        self.bits = bits
        self.observer = observer
        self.q_type = 0
        self.reduce_range = reduce_range

        self.bit_scale = bit_scale
        # scale/zero_point/eps
        self.use_gpu = config.use_gpu
        if self.use_gpu:
            self.device = config.device
        else:
            self.device = 'cpu'
        # if ((config.qat_lstm) or (config.ma_lstm)):
        #     self.register_buffer("scale", torch.ones((1), dtype=torch.float16))
        # else:
        self.register_buffer("scale", torch.ones((1), dtype=torch.float32))
        self.register_buffer("zero_point", torch.zeros((1), dtype=torch.int))
        self.eps = torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32, device= self.device)

    def update_qparams(self):
        raise NotImplementedError

    # 取整(ste)
    def round(self, input, observer_min_val, observer_max_val, q_type):
        output = Round.apply(input, observer_min_val, observer_max_val, q_type)
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            print("！Binary quantization is not supported ！")
            assert self.bits != 1
        else:

            # qat, update quant_para
            if self.training:
                self.observer(input)  # update observer_min and observer_max
                self.update_qparams()  # update scale and zero_point

            self.zero_point = self.zero_point.to(input.device)
            self.scale = self.scale.to(input.device)
            output = (
                             torch.clamp(
                                 self.round(
                                     input / self.scale.clone() + self.zero_point,
                                     self.observer.min_val / self.scale + self.zero_point,
                                     self.observer.max_val / self.scale + self.zero_point,
                                     self.q_type,
                                 ),
                                 self.quant_min_val,
                                 self.quant_max_val,
                             )
                             - self.zero_point
                     ) * self.scale.clone()

        return output

class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(SignedQuantizer, self).__init__(*args, **kwargs)

        self.quant_min_val = torch.tensor((-((1 << (self.bits - 1)) - 1)), dtype=torch.float32, device= self.device)


        self.quant_max_val = torch.tensor(((1 << (self.bits - 1)) - 1), dtype=torch.float32, device= self.device)

class UnsignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super(UnsignedQuantizer, self).__init__(*args, **kwargs)
        self.quant_min_val = torch.tensor((0), dtype=torch.float32, device= self.device)
        if not self.reduce_range:
            self.quant_max_val = torch.tensor(((1 << self.bits) - 1), dtype=torch.float32, device= self.device)
        else:
            self.quant_max_val = torch.tensor((1 << (self.bits-1)), dtype=torch.float32, device=self.device)

# 对称量化
class SymmetricQuantizer(SignedQuantizer):
    def update_qparams(self):
        self.q_type = 0
        quant_range = (
                float(self.quant_max_val - self.quant_min_val) / 2
        )  # quantized_range
        float_range = torch.max(
            torch.abs(self.observer.min_val), torch.abs(self.observer.max_val)
        )  # float_range
        scale = float_range / quant_range  # scale
        scale = torch.max(scale, self.eps)  # processing for very small scale
        zero_point = torch.zeros_like(scale)  # zero_point
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)


# 非对称量化
class AsymmetricQuantizer(UnsignedQuantizer):
    def update_qparams(self):
        self.q_type = 1
        quant_range = float(self.quant_max_val - self.quant_min_val)  # quantized_range
        float_range = self.observer.max_val - self.observer.min_val  # float_range
        scale = float_range / quant_range  # scale
        scale = torch.max(scale, self.eps)  # processing for very small scale
        sign = torch.sign(self.observer.min_val)
        zero_point = - sign * torch.floor(
            torch.abs(self.observer.min_val / scale) + 0.5
        )  # zero_point
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)
