# WideLinears
## _Pytorch parallel Neural Networks_

A package of pytorch modules for fast paralellization of separate deep neural networks.
Ideal for agent-based systems such as evolutionary algorithms.

## Installation

WideLinear is avaliable through [Pypi](https://pypi.org/project/widelinears/)


```sh
pip install widelinears
```


## Pytorch Modules

### WideLinear
Represents a family of parallel Linear layers that share the same input and output sizes

##### Parameters
- __beings__ (int): Number of parallel Linear layers
- __input_size__ (int): Size of input of each linear layer
- __output_size__ (int): Size of output of each linear layer

##### Input Tensor Shapes
- __(input_size,)__ will clone this input and give it to each Linear in the module, outputs __(beings, output_size)__
- __(beings, input_size)__ will give each Linear its own input vector, outputs __(beings, output_size)__
- __(batch, beings, input_size)__ will give each Linear its own batch of inputs, outputs __(batch, beings, output_size)__

##### Methods
- __forward__ (Tensor): Returns output for input tensors as explained above
- __clone_being__ (source, destination): Clones linear layer from one position to other, overriding what was there
- __get_single__ (position): Get __LinearWidePointer__ class that is a pointer to this module but behaves as a normal single __nn.Linear__
- __to_linears__ (): Returns list of instances of __nn.Linear__ with the same parameters as each Linear ins this module

### WideDeep
WideDeep generalizes Deep Neural Networks using WideLinear layers, and simplifies constructing parallel Deep Neural Networks. Behaves as a group of separate Deep Neural Networks that run in parallel for good time efficiency.

##### Parameters
- __beings__ (int): Number of parallel Deep NNs
- __input_size__ (int): Size of input of each Deep NN
- __hidden_size__ (int): Size of each hidden layer in each Deep NN
- __depth__ (int): Number of hidden layers (if 1, there is a single Linear layer from input to output)
- __output_size__ (int): Size of output of each Deep NN
- __non_linear__ (optional function): Non Linearity function at each intermediate step (defaults to ReLU)
- __final_nl__ (optional function): Non Linearity at output (defaults to sigmoid)

##### Input Tensor Shapes
- __(input_size,)__ will clone this input and give it to each Deep NN, outputs __(beings, output_size)__
- __(beings, input_size)__ will give each Deep NN its own input vector, outputs __(beings, output_size)__
- __(batch, beings, input_size)__ will give each Deep NN its own batch of inputs, outputs __(batch, beings, output_size)__

##### Methods
- __forward__ (Tensor): Returns output for input tensors as explained above
- __clone_being__ (source, destination): Clones Deep NN from one position to other, overriding what was there

![Model diagram](https://i.ibb.co/KbdD83S/deep-diag.png)
__Example architecture__ for parameters:
- __beings__ = 4
- __input_size__ = 5
- __hidden_size__ = 3
- __depth__ = 3
- __output_size__ = 4



## License

MIT

Made by João Figueira


