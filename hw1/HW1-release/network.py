#========================================================
#             Media and Cognition
#             Homework 1 Neural network basics
#             network.py - linear layer and MLP network
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2024
#========================================================
import torch
import torch.nn as nn
from activations import Activation

'''
In this script we will implement our Linear layer and MLP network.
For the linear layer, we will provide a sample of codes which calculate both the forward and backward processes by our own.
More details about customizing a backward process can be found in:
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
For the MLP network, you should cascade the linear layers and activation functions in a proper way in the __init__ function and implement the forward function.
'''


class LinearFunction(torch.autograd.Function):
    '''
    we will implement the linear function:
    y = xW^T + b
    as well as its gradient computation process
    '''

    @staticmethod
    def forward(ctx, x, W, b):
        '''
        Input:
        :param ctx: a context object that can be used to stash information for backward computation
        :param x: input features with size [batch_size, input_size]
        :param W: weight matrix with size [output_size, input_size]
        :param b: bias with size [output_size]
        Return:
        y :output features with size [batch_size, output_size]
        '''

        y = torch.matmul(x, W.T) + b
        ctx.save_for_backward(x, W)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        '''
        Input:
        :param ctx: a context object with saved variables
        :param grad_output: dL/dy, with size [batch_size, output_size]
        Return:
        grad_input: dL/dx, with size [batch_size, input_size]
        grad_W: dL/dW, with size [output_size, input_size], summed for data in the batch
        grad_b: dL/db, with size [output_size], summed for data in the batch
        '''

        x, W = ctx.saved_variables

        # calculate dL/dx by using dL/dy (grad_output) and W, e.g., dL/dx = dL/dy*W
        # calculate dL/dW by using dL/dy (grad_output) and x
        # calculate dL/db using dL/dy (grad_output)
        # you can use torch.matmul(A, B) to compute matrix product of A and B

        grad_input = torch.matmul(grad_output, W)
        grad_W = torch.matmul(grad_output.T, x)
        grad_b = grad_output.sum(0)

        return grad_input, grad_W, grad_b


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        '''
        A linear layer which uses our own LinearFunction implemented above.
        -----------------------------------------------
        :param input_size: dimension of input features
        :param output_size: dimension of output features
        '''
        super(Linear, self).__init__()

        W = torch.randn(output_size, input_size).float()
        b = torch.zeros(output_size).float()
        self.W = nn.Parameter(W, requires_grad=True)
        self.b = nn.Parameter(b, requires_grad=True)

    def forward(self, x):
        # here we call the LinearFunction we implement above
        return LinearFunction.apply(x, self.W, self.b)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, act_type):
        '''
        Multilayer Perceptron
        ----------------------
        :param input_size: dimension of input features
        :param output_size: dimension of output features
        :param hidden_size: a list containing hidden size for each hidden layer
        :param n_layers: number of layers
        :param act_type: type of activation function for each hidden layer, can be none, sigmoid, tanh, or relu
        '''
        # TODO 1: initialize the parent class nn.Module
        ???

        # total layer number should be hidden layer number + 1 (output layer)
        assert len(hidden_size) + 1 == n_layers, 'total layer number should be hidden layer number + 1'

        # TODO 2；complete the network structures 
        # instantiate the activation function by using the defined classes in activations.py
        ???

        # initialize a list to save layers
        layers = nn.ModuleList()

        if n_layers == 1:
            # append a linear layer into the module list
            # if n_layers == 1, MLP degenerates to a single linear layer
            ???

        # MLP with at least 2 layers
        else:
            # construct the hidden layers and add them to the module list
            # a hidden layer of MLP consists of a linear layer and an activation function
            in_size = input_size
            for i in range(n_layers - 1):
                layer = Linear(in_size, hidden_size[i])
                ??? # append the linear layer into the module list
                layers.append(self.act)
                ??? # update in_size for the next layer

            # initialize the output layer and append the layer into the module list
            # hint: what is the output size of the output layer?
            ???

        # Use nn.Sequential to get the neural network
        ???


    def forward(self, x):
        '''
        Define the forward function
        :param x: input features with size [batch_size, input_size]
        :return: output features with size [batch_size, output_size]
        '''
        # TODO 3: implement the forward propagation of the MLP
        out = ???

        return out
