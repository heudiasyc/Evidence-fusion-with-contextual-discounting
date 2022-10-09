# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export
from torch.nn.parameter import Parameter
import numpy as np
__all__ = ["UNet_DS_t2", "Unet_ds_t2"]


@export("monai.networks.nets")
@alias("Unet_ds_t2")
class DsFunction1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, W, BETA, alpha, gamma):
        class_dim = 4
        prototype_dim =20
        [batch_size, in_channel,height,weight,depth] = input.size()

        BETA2 = BETA * BETA
        beta2 = BETA2.t().sum(0)
        U = BETA2 / (beta2.unsqueeze(1) * torch.ones(1, class_dim,device=input.device))
        alphap = 0.99 / (1 + torch.exp(-alpha))


        d = torch.zeros(prototype_dim, batch_size,height,weight,depth,device=input.device)
        s = torch.zeros(prototype_dim, batch_size,height,weight,depth,device=input.device)
        expo = torch.zeros(prototype_dim, batch_size,height,weight,depth,device=input.device)

        mk = torch.cat((torch.zeros(class_dim, batch_size,height,weight,depth, device=input.device),torch.ones(1, batch_size, height,weight,depth,device=input.device)), 0)

        for k in range(prototype_dim):
            temp = input.permute(1,0,2,3,4) - torch.mm(W[k, :].unsqueeze(1), torch.ones(1, batch_size, device=input.device)).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            d[k, :] = 0.5 * (temp * temp).sum(0)
            expo[k, :] = torch.exp(-gamma[k] ** 2 * d[k, :])
            s[k, :] = alphap[k] * expo[k, :]
            m = torch.cat((U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4) * s[k, :], torch.ones(1, batch_size, height,weight,depth,device=input.device) - s[k, :]), 0)

            t2 = mk[:class_dim, :, :, :,:] * (m[:class_dim, :, :, :,:] + torch.ones(class_dim, 1, height, weight,depth, device=input.device)* m[class_dim, :, :, :,:])
            t3 = m[:class_dim, :, :, :,:] * (torch.ones(class_dim, 1, height, weight, depth,device=input.device)* mk[class_dim, :, :, :,:])
            t4 = (mk[class_dim, :,:,:,:]) * (m[class_dim, :,:,:,:]).unsqueeze(0)
            mk = torch.cat((t2 + t3, t4), 0)

        K = mk.sum(0)
        mk_n = (mk / (torch.ones(class_dim + 1, 1,height,weight,depth, device=input.device) * K)).permute(1,0,2,3,4)
        #mass_b = mk_n[:, :class_dim,:,:,:] + (1 / class_dim) * mk_n[:, class_dim,:,:,:].unsqueeze(1)# * torch.ones(1, class_dim,height,weight,depth,device=input.device)
        ctx.save_for_backward(input, W, BETA, alpha, gamma, mk, d)

        return mk_n

    @staticmethod
    def backward(ctx, grad_output):

        input, W, BETA, alpha, gamma, mk, d = ctx.saved_tensors
        grad_input = grad_W = grad_BETA = grad_alpha = grad_gamma = None

        M = 4
        prototype_dim = 20
        [batch_size, in_channel,height,weight,depth] = input.size()
        mu = 0  # regularization parameter (default=0)
        iw = 1  # 1 if optimization of prototype centers, 0 otherwise (default=1)
        grad_output_ = grad_output[:,:4,:,:,:]*batch_size*M*height*weight*depth

        K = mk.sum(0).unsqueeze(0)
        K2 = K ** 2
        BETA2 = BETA * BETA
        beta2 = BETA2.t().sum(0).unsqueeze(1)
        U = BETA2 / (beta2 * torch.ones(1, M,device=input.device))
        alphap = 0.99 / (1 + torch.exp(-alpha))  # 200*1
        I = torch.eye(M, device=grad_output.device)

        s = torch.zeros(prototype_dim, batch_size,height,weight,depth,device=input.device)
        expo = torch.zeros(prototype_dim, batch_size,height,weight,depth,device=input.device)
        mm = torch.cat((torch.zeros(M, batch_size,height,weight,depth,device=input.device),torch.ones(1, batch_size,height,weight, depth,device=input.device)), 0)

        dEdm = torch.zeros(M + 1, batch_size, height,weight, depth,device=input.device)
        #dEdm_test = torch.zeros(M + 1, batch_size, height, weight, depth,device=input.device)
        dU = torch.zeros(prototype_dim, M,device=input.device)
        Ds = torch.zeros(prototype_dim, batch_size, height,weight,depth,device=input.device)
        DW = torch.zeros(prototype_dim, in_channel,device=input.device)

        for p in range(M):

            dEdm[p, :] = (grad_output_.permute(1, 0, 2, 3,4) * (
                        I[:, p].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4) * K - mk[:M, :] - 1 / M * (
                            torch.ones(M, 1, height, weight, depth,device=input.device) * mk[M, :]))).sum(0) / K2

        dEdm[M, :] = ((grad_output_.permute(1, 0, 2, 3,4) * (
                    - mk[:M, :] + 1 / M * torch.ones(M, 1, height, weight,depth, device=input.device) * (K - mk[M, :]))).sum(
            0)) / K2
        #dEdm_t[M, :] = ((grad_output_t.t() * (- mk_t[:M, :] + 1 / 3 * torch.ones(3, 1, device=input.device) * (K_t - mk_t[M, :]))).sum(0)) / K2_t

        for k in range(prototype_dim):
            expo[k, :] = torch.exp(-gamma[k] ** 2 * d[k, :])
            s[k] = alphap[k] * expo[k, :]
            m = torch.cat((U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4) * s[k, :],torch.ones(1, batch_size, height, weight,depth, device=input.device) - s[k, :]), 0)
            mm[M, :] = mk[M, :] / m[M, :]
            L = torch.ones(M, 1,height,weight,depth,device=input.device) * mm[M, :]    # L:m_M+1
            mm[:M, :] = (mk[:M, :] - L * m[:M, :]) / (m[:M, :] + torch.ones(M, 1,height,weight,depth,device=input.device) * m[M, :])  # m_j
            R = mm[:M, :] + L     # function 97,
            A = R * torch.ones(M, 1,height,weight,depth,device=input.device) * s[k, :]  # function 97, s
            B = U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4) * torch.ones(1, batch_size, height,weight,depth,device=input.device) * R - mm[:M, :]
            #tet=torch.mean((A * dEdm[:M, :]).view(3,-1).permute(1,0),0)
            dU[k, :] = torch.mean((A * dEdm[:M, :]).view(M,-1).permute(1,0),0)
            Ds[k, :] = (dEdm[:M, :] * B).sum(0) - (dEdm[M, :] * mm[M, :])
            #test=Ds[k, :] * gamma[k] ** 2 * s[k, :]
            tt1 = Ds[k, :] * (gamma[k] ** 2 * torch.ones(1, batch_size, height, weight, depth,device=input.device)) * s[k, :]
            tt2 = (torch.ones(batch_size, 1, device=input.device) * W[k, :]).unsqueeze(2).unsqueeze(
                3).unsqueeze(4) - input  # - input
            tt1 = tt1.view(1, -1)
            tt2 = tt2.permute(1,0,2,3,4).reshape(in_channel, batch_size*height*weight*depth).permute(1,0)
            DW[k, :] = -torch.mm(tt1, tt2)



        DW = iw * DW / (batch_size*height*weight*depth)
        T = beta2 * torch.ones(1, M, device=input.device)
        Dbeta = (2 * BETA / T ** 2) * (dU * (T - BETA2) - (dU * BETA2).sum(1).unsqueeze(1) * torch.ones(1, M,
                                                                                                        device=input.device) + dU * BETA2)
        Dgamma = - 2 * torch.mean(((Ds * d * s).view(prototype_dim, -1)).t(), 0).unsqueeze(1) * gamma
        Dalpha = (torch.mean(((Ds * expo).view(prototype_dim, -1)).t(), 0).unsqueeze(1) + mu) * (
                    0.99 * (1 - alphap) * alphap)
        Dinput = torch.zeros(batch_size, in_channel,height,weight,depth,device=input.device)
        temp2 = torch.zeros(prototype_dim, in_channel,height,weight,depth, device=input.device)

        for n in range(batch_size):
            for k in range(prototype_dim):
                test7=input[n, :] - W[k, :].unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                test9= (Ds[k, n,:,:] * (gamma[k] ** 2) * s[k, n,:,:]).unsqueeze(0).unsqueeze(1)
                temp2[k]=-prototype_dim*test9*test7
                Dinput[n, :] = temp2.mean(0)


        if ctx.needs_input_grad[0]:
            grad_input = Dinput
        if ctx.needs_input_grad[1]:
            grad_W = DW
        if ctx.needs_input_grad[2]:
            grad_BETA = Dbeta
        if ctx.needs_input_grad[3]:
            grad_alpha = Dalpha
        if ctx.needs_input_grad[4]:
            grad_gamma = Dgamma

        return grad_input, grad_W, grad_BETA, grad_alpha, grad_gamma

class Ds1(nn.Module):
    def __init__(self,input_dim, prototpye_dim, class_dim):
        super(Ds1, self).__init__()
        self.input_dim = input_dim
        self.class_dim = class_dim
        self.prototype_dim = prototpye_dim
        self.BETA = Parameter(torch.Tensor(self.prototype_dim, self.class_dim))
        self.alpha = Parameter(torch.Tensor(self.prototype_dim, 1))
        self.gamma = Parameter(torch.Tensor(self.prototype_dim, 1))
        self.W = Parameter(torch.Tensor(self.prototype_dim, self.input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        #with torch.no_grad():
        #    self.W.copy_(W_p)
        nn.init.normal_(self.W)
        nn.init.xavier_uniform_(self.BETA)
        nn.init.constant_(self.gamma, 0.1)
        nn.init.constant_(self.alpha, 0)

    def forward(self, input):
        return DsFunction1.apply(input, self.W, self.BETA, self.alpha, self.gamma)


class UNet_DS_t2(nn.Module):

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Sequential:
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            return nn.Sequential(down, SkipConnection(subblock), up)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

        #for p in self.model.parameters():
        #    p.requires_grad=False
        self.ds1 = Ds1(4,20,4)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x=self.ds1(x)
        return x


Unet_ds_t2 = unet_ds_t2 = UNet_DS_t2
