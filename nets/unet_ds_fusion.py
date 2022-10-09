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
from monai.networks.nets import UNet_DS_flair,UNet_DS_t1,UNet_DS_t2,UNet_DS_t1ce
import numpy as np
__all__ = ["UNet_DS_fusion", "Unet_ds_fusion"]


@export("monai.networks.nets")
@alias("Unet_ds_fusion")

class Fusion(nn.Module):
    def __init__(self, class_dim):
        super(Fusion, self).__init__()
        self.class_dim=class_dim

        self.alpha1 = Parameter(torch.Tensor(self.class_dim,1,1,1))
        self.alpha2 = Parameter(torch.Tensor(self.class_dim,1,1,1))
        self.alpha3 = Parameter(torch.Tensor(self.class_dim,1,1,1))
        self.alpha4 = Parameter(torch.Tensor(self.class_dim,1,1,1))
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.constant_(self.alpha1, 0)
        nn.init.constant_(self.alpha2, 0)
        nn.init.constant_(self.alpha3, 0)
        nn.init.constant_(self.alpha4, 0)


    def forward(self, input1,input2,input3,input4):
        [batch_size, class_dim, height, weight, depth] = input1.size()
        x1 = input1[:, :4, :, :, :] +input1[:, 4, :, :, :].unsqueeze(1)
        x2 = input2[:, :4, :, :, :] +input2[:, 4, :, :, :].unsqueeze(1)
        x3 = input3[:, :4, :, :, :] +input3[:, 4, :, :, :].unsqueeze(1)
        x4= input4[:, :4, :, :, :] +input4[:, 4, :, :, :].unsqueeze(1)

        alpha1 = 1 / (1 + torch.exp(-self.alpha1))
        alpha2 = 1 / (1 + torch.exp(-self.alpha2))
        alpha3 = 1 / (1 + torch.exp(-self.alpha3))
        alpha4 = 1 / (1 + torch.exp(-self.alpha4))
        batch=torch.ones(batch_size, class_dim-1, height, weight, depth,device=input1.device)
        alpha1=batch*alpha1
        alpha2 = batch * alpha2
        alpha3 = batch * alpha3
        alpha4 = batch * alpha4



        a_x1=alpha1+(1-alpha1)*x1
        a_x2 = alpha2 + (1 - alpha2) * x2
        a_x3 = alpha3 + (1 - alpha3) * x3
        a_x4 = alpha4 + (1 - alpha4) * x4
        pl=a_x1*a_x2*a_x3*a_x4
        K = pl.sum(1)
        pl = (pl / (torch.ones(batch_size, class_dim-1, height, weight, depth, device=x1.device) * K.unsqueeze(1)))
        return pl

class UNet_DS_fusion(nn.Module):

    def __init__(self):
        super(UNet_DS_fusion, self).__init__()
        self.t1ce_modality =  UNet_DS_t1ce(
        dimensions=3,  # 3D
        in_channels=1,
        out_channels=4,
        kernel_size=5,
        #channels=(8, 16, 32, 64, 128),
        channels=(16, 32, 64, 128,256),
        strides=(2, 2, 2, 2),
        num_res_units=2,)
        self.t1_modality =  UNet_DS_t1(
        dimensions=3,  # 3D
        in_channels=1,
        out_channels=4,
        kernel_size=5,
        #channels=(8, 16, 32, 64, 128),
        channels=(16, 32, 64, 128,256),
        strides=(2, 2, 2, 2),
        num_res_units=2,)
        self.flair_modality =  UNet_DS_flair(
        dimensions=3,  # 3D
        in_channels=1,
        out_channels=4,
        kernel_size=5,
        #channels=(8, 16, 32, 64, 128),
        channels=(16, 32, 64, 128,256),
        strides=(2, 2, 2, 2),
        num_res_units=2,)
        self.t2_modality =  UNet_DS_t2(
        dimensions=3,  # 3D
        in_channels=1,
        out_channels=4,
        kernel_size=5,
        #channels=(8, 16, 32, 64, 128),
        channels=(16, 32, 64, 128,256),
        strides=(2, 2, 2, 2),
        num_res_units=2,)
        for p in self.parameters():
            p.requires_grad=False
        self.fusion = Fusion(4)


    def forward(self, x):
        x1=x[:,0,].unsqueeze(1)
        x2 = x[:, 1, ].unsqueeze(1)
        x3 = x[:, 2, ].unsqueeze(1)
        x4 = x[:, 3, ].unsqueeze(1)

        x11 = self.t1ce_modality(x1)
        x22=self.t1_modality(x2)
        x33=self.flair_modality(x3)
        x44=self.t2_modality(x4)

        x = self.fusion(x11,x22,x33,x44)
        return x



Unet_ds_fusion = unet_ds_fusion = UNet_DS_fusion
