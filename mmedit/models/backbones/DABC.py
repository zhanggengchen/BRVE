import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb
    

class DACA(nn.Module):
    def __init__(self, channels):
        super(DACA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channels, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(3, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, eps=1e-6):
        b, c, h, w = x.shape
        channel_mean = x.view(b, c, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(b, c, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        channel_abs = self.avg_pool(torch.abs(x)).squeeze(-1)

        y = torch.cat([channel_abs, channel_mean, channel_std], dim=2)

        y = self.conv(y.transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return self.sigmoid(y)


class RPReLU(nn.Module):
    def __init__(self, inplanes):
        super(RPReLU, self).__init__()
        self.pr_bias0 = LearnableBias(inplanes)
        self.pr_prelu = nn.PReLU(inplanes)
        self.pr_bias1 = LearnableBias(inplanes)

    def forward(self, x):
        x = self.pr_bias1(self.pr_prelu(self.pr_bias0(x)))
        return x

class BinaryQuantize_Quad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
    
    def unsupported_flops(self, module, inputs, output):
        input = inputs[0]
        inputs_dims = list(input.shape)
        overall_flops = int(np.prod(inputs_dims))

        module.__flops__ += overall_flops


class HardBinaryConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1,bias=True):
        super(HardBinaryConv, self).__init__(
            in_chn,
            out_chn,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0) # STE
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights,self.bias, stride=self.stride, padding=self.padding)

        return y
    
    def unsupported_flops(self, module, inputs, output):
        # 补充 BNN flops 计算，与conv2d相同
        input = inputs[0] # input = (input, fea)
        conv_per_position_flops = self.kernel_size[0] * self.kernel_size[1] * self.in_channels * self.out_channels // self.groups
        batch_size = input.shape[0]
        output_dims = list(output.shape[2:])
        active_elements_count = batch_size * int(np.prod(output_dims))
        overall_conv_flops = conv_per_position_flops * active_elements_count

        bias_flops = 0
        if self.bias is not None:
            bias_flops = self.out_channels * active_elements_count
        overall_flops = (overall_conv_flops // 64) + bias_flops

        module.__flops__ += overall_flops
    

class DABCConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(DABCConv2d, self).__init__()

        self.move0 = LearnableBias(in_channels)
        self.binary_conv = HardBinaryConv(in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size//2),
        bias=bias)
        self.relu=RPReLU(out_channels)
        self.daca = DACA(out_channels)

    def forward(self, x):
        out = self.move0(x)
        scale = self.daca(out)
        out = BinaryQuantize_Quad().apply(out)
        out = self.binary_conv(out)
        out = out * scale
        out = self.relu(out)
        out = out + x
        return out


class BinaryConv2dSkip1x1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=False):
        super(BinaryConv2dSkip1x1, self).__init__()

        self.move0 = LearnableBias(in_channels)
        self.binary_conv = HardBinaryConv(in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size//2),
        bias=bias)
        self.relu=RPReLU(out_channels)
        self.conv_skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=groups)

    def forward(self, x):
        out = self.move0(x)
        out = BinaryQuantize_Quad().apply(out)
        out = self.binary_conv(out)
        out =self.relu(out)
        out = out + self.conv_skip(x)
        return out


class BNNDownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.move0 = LearnableBias(in_channels)
        self.binary_conv = HardBinaryConv(in_channels, out_channels, 
                                          kernel_size=3, stride=2, padding=1)
        self.relu=RPReLU(out_channels)

        self.conv_skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.pooling = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.move0(x)
        out = BinaryQuantize_Quad().apply(out)
        out = self.binary_conv(out)
        out = self.relu(out)
        out = out + self.conv_skip(self.pooling(x))

        return out


class BNNUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = BinaryConv2dSkip1x1(in_channels, out_channels, kernel_size=3)

    def forward(self, x):
        out = self.conv(self.up(x))

        return out


class BNNSkipUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = BinaryConv2dSkip1x1(in_channels, out_channels, kernel_size=3)

    def forward(self, x, y):
        out = self.conv(self.up(x))

        return out + y