import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size,stride,padding,dilation=1,groups=1, inference_mode=False,num_conv_branches=1):
        super(ConvBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.activation = nn.GELU()
        self.se = nn.Identity()
        self.dropout = nn.Dropout(0.2)

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            # Re-parameterizable skip connection
            self.BN_skip = (nn.BatchNorm2d(num_features=in_channels) if in_channels == out_channels and stride==1 else None)
            # Re-parameterizable conv branches
            if num_conv_branches>0:
                Conv1=list()
                for _ in range(self.num_conv_branches):
                    Conv1.append(self.conv_bn(kernel_size,padding))
                self.Conv1 = nn.ModuleList(Conv1)
            else:
                self.Conv1 = None

            # Re-parameterizable scale branch
            self.Conv2 = None
            if (kernel_size > 1):
                self.Conv2 = self.conv_bn(kernel_size=1, padding=0)

    def forward(self,x):
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        BN=0
        if self.BN_skip is not None:
            BN = self.BN_skip(x)

        conv2=0
        if self.Conv2 is not None:
            conv2 = self.Conv2(x)

        out=BN+conv2
        if self.Conv1 is not None:
            for i in range(self.num_conv_branches):
                out += self.Conv1[i](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        if self.inference_mode:
            return
        kernel,bias =self.get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__("Conv1")
        self.__delattr__("Conv2")
        if hasattr(self, "BN_skip"):
            self.__delattr__("BN_skip")

        self.inference_mode = True

    def get_kernel_bias(self):
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self.fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self.fuse_bn_tensor(self.rbr_conv[ix])
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def fuse_bn_tensor(self, branch):
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def conv_bn(self, kernel_size: int, padding: int):
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list