# Copyright 2022 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.unfold_0 = nn.Unfold(kernel_size=3)
        self.unfold_1 = nn.Unfold(kernel_size=(2,4), stride=(2,1), padding=2, dilation=1)
        self.unfold_2 = nn.Unfold(kernel_size=(1,3), stride=1, padding=(2,4), dilation=(1,2))

    def forward(self, x):
        x0 = self.unfold_0(x)
        x1 = self.unfold_1(x)
        x2 = self.unfold_2(x)

        return x0, x1, x2

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 64, 64)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_Unfold.pt")

    # torchscript to ncnn
    import os
    os.system("../../src/pnnx test_nn_Unfold.pt inputshape=[1,12,64,64]")

    # ncnn inference
    import test_nn_Unfold_ncnn
    b = test_nn_Unfold_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.allclose(a0, b0, 1e-4, 1e-4):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
