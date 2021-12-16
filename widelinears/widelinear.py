import math
import torch
from torch.nn import Linear
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


# pointer to one line in a widelinear, behaves like normal linear module
class LinearWidePointer(nn.Module):
    def __init__(self, hive, being:int):
        super(LinearWidePointer, self).__init__()

        self.hive = hive
        self.being = being

    def forward(self, x: Tensor) -> Tensor:

        if len(x.shape) == 1: # (input,)
            x = torch.matmul(x, self.hive.weight[self.being])
            x = x + self.hive.bias[self.being]

        elif len(x.shape) == 2: # (batch, input)
            x = torch.matmul(x, self.hive.weight[self.being])
            x = x + self.hive.bias[self.being]
        
        else:
            raise ValueError("wrong input shape")

        return x

    # convert to normal nn.Linear
    def to_linear(self) -> Linear:

        linear = Linear(self.hive.input_size, self.hive.output_size)
        linear.weight.data = self.hive.weight.data[self.being].t()
        linear.bias.data = self.hive.bias.data[self.being]

        return linear

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.hive.input_size, self.hive.output_size, self.hive.bias is not None
        )

# Family of separate Linear layers that run in parallel
class WideLinear(nn.Module):
    def __init__(self, beings: int, input_size: int, output_size: int):
        super(WideLinear, self).__init__()

        self.beings = beings
        self.input_size = input_size
        self.output_size = output_size

        # Initialization based on code from nn.Linear
        weight = nn.Parameter(torch.empty((beings, input_size, output_size)))
        bias = nn.Parameter(torch.empty((beings, output_size)))
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(output_size) if output_size > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

        self.weight = weight
        self.bias = bias


    def forward(self, x: Tensor) -> Tensor:

        if len(x.shape) == 1: # (input,)
            x = x.repeat((self.beings, 1))
            x = torch.einsum("bi,bio->bo", x, self.weight)
            x = x + self.bias

        elif len(x.shape) == 2: # (beings, input)
            x = torch.einsum("bi,bio->bo", x, self.weight)
            x = x + self.bias

        elif len(x.shape) == 3: # (batch, beings, input)
            batchsize = x.shape[0]
            x = torch.einsum("xbi,bio->xbo", x, self.weight)
            x = x + self.bias.repeat(batchsize, 1, 1)
        
        else:
            raise ValueError("wrong input shape")

        return x

    # clone linear to outher position
    def clone_being(self, source: int, destination: int):
        self.weight.data[destination] = self.weight.data[source]
        self.bias.data[destination] = self.bias.data[source]

    # get module that behaves like single linear layer
    def get_single(self, being: int) -> LinearWidePointer:
        return LinearWidePointer(self, being)

    def extra_repr(self) -> str:
        return 'beings={}, in_features={}, out_features={}, bias={}'.format(
            self.beings, self.input_size, self.output_size, self.bias is not None
        )

    # return list of nn.Linear modules
    def to_linears(self):
        linears = []
        for i in range(self.beings):
            linears += [self.get_single(i).to_linear()]
        return linears


if __name__ == "__main__":
    # run some tests
    print("testing module...")
    beings = 100
    insize = 3
    outsize = 4
    batch = 2
    hv = WideLinear(beings, insize, outsize)
    print("Hive {}x {}->{}".format(beings, insize, outsize))
    input1 = torch.randn((insize))
    input2 = torch.randn((beings, insize))
    input3 = torch.randn((batch, beings, insize))
    output1 = hv(input1)
    output2 = hv(input2)
    output3 = hv(input3)
    print("output1 shape {}, expected {}".format(output1.shape, (beings, outsize)))
    print("output2 shape {}, expected {}".format(output2.shape, (beings, outsize)))
    print("output3 shape {}, expected {}".format(output3.shape, (batch, beings, outsize)))
    input4 = torch.randn((beings, insize+1))
    input5 = torch.randn((beings+1, insize))
    try:
        output4 = hv(input4)
    except:
        print("incorrect input threw exception as expected")
    try:
        output5 = hv(input5)
    except:
        print("incorrect input threw exception as expected")
    being = 1
    # test single layers
    pointer = hv.get_single(being)
    linear = pointer.to_linear()
    print(pointer)
    print(linear)
    input1_s = input1
    input2_s = input2[being]
    input3_s = input3[:, being, :]
    output1_p = pointer(input1_s)
    output2_p = pointer(input2_s)
    output3_p = pointer(input3_s)
    output1_l = linear(input1_s)
    output2_l = linear(input2_s)
    output3_l = linear(input3_s)
    print("Testing output 1:")
    print(output1[being])
    print(output1_p)
    print(output1_l)
    print("Testing output 2:")
    print(output2[being])
    print(output2_p)
    print(output2_l)
    print("Testing output 3:")
    print(output3[:, being, :])
    print(output3_p)
    print(output3_l)
    # test cuda
    hv = hv.to("cuda")
    inputcuda = input2.cuda()
    outputcuda = hv(inputcuda)
    print("outputcuda shape {}, expected {}".format(outputcuda.shape, (beings, outsize)))
    inputcuda_s = inputcuda[being]
    outputcuda_s = pointer(inputcuda_s)
    print("Testing output cuda:")
    print(outputcuda[being])
    print(outputcuda_s)
    print("Testing to_linears")
    linears = hv.to_linears()
    print("got {} nn.Linear".format(len(linears)))
    print("Done testing")
