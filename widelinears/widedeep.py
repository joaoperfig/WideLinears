import math
import torch
from torch.nn import Linear
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from widelinears.widelinear import WideLinear, LinearWidePointer

# Family of separate deep NNs for very fast paralel forward passes
class WideDeep(nn.Module):
    def __init__(self, beings: int, input_size: int, hidden_size: int, depth: int, output_size: int, non_linear=None, final_nl=None):
        super(WideDeep, self).__init__()

        self.beings = beings
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_size = output_size

        if non_linear is None:
            self.non_linear = nn.ReLU()
        else:
            self.non_linear = non_linear
        if final_nl is None:
            self.final_nl = nn.Sigmoid()
        else:
            self.final_nl = final_nl

        linears = []

        lastsize = input_size
        for nextsize in (([hidden_size]*(depth)) + [output_size]):

            linear = WideLinear(beings, lastsize, nextsize)
            linears += [linear]
            lastsize = nextsize
            
        self.linears = nn.ModuleList(linears)


    def forward(self, x):

        if len(x.shape) == 1: # (input,)
            for i, linear in enumerate(self.linears):
                x = linear(x)
                if i < self.depth:
                    x = self.non_linear(x)
                else:
                    x = self.final_nl(x)

        elif len(x.shape) == 2: # (beings, input)
            for i, linear in enumerate(self.linears):
                x = linear(x)
                if i < self.depth:
                    x = self.non_linear(x)
                else:
                    x = self.final_nl(x)


        elif len(x.shape) == 3: # (batch, beings, input)
            for i, linear in enumerate(self.linears):
                x = linear(x)
                if i < self.depth:
                    x = self.non_linear(x)
                else:
                    x = self.final_nl(x)

        return x


    def clone_being(self, source, destination):
        for linear in self.linears:
            linear.clone_being(source, destination)


if __name__ == "__main__":
    # run some tests
    beings = 100
    input_size = 5
    hidden_size = 3
    depth = 2
    output_size = 4
    hd = WideDeep(beings, input_size, hidden_size, depth, output_size)
    print(hd)
    input = torch.randn((beings, input_size))
    output = hd(input)
    print("output shape {}, expected {}".format(output.shape, (beings, output_size)))
    # test cuda
    hd = hd.to("cuda")
    inputc = torch.randn((beings, input_size)).cuda()
    outputc = hd(inputc)
    print("output shape {}, expected {}".format(outputc.shape, (beings, output_size)))
    print(outputc)
    print("Done testing")
