#========================================================
#             Media and Cognition
#             Homework 1 Neural network basics
#             activations.py - activation functions
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2024
#========================================================
import torch
import torch.nn as nn

'''
In this script we will implement three activation functions, including both forward and backward processes.
More details about customizing a backward process in PyTorch can be found in:
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
'''

## Here, Tanh is given as an example to show how to construct the activation function. Please finish the activation functions of Sigmoid and ReLU later.
class Tanh(torch.autograd.Function):
    '''
    Tanh activation function
    y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    '''
    # static method of a python class means that we can call the function without initializing an instance of the class
    @staticmethod
    def forward(ctx, x):
        '''
        In the forward pass we receive a Tensor containing the input x and return
        a Tensor containing the output. 
        
        ctx: it is a context object that can be used to save information for backward computation. You can save 
        objects by using ctx.save_for_backward, and get objects by using ctx.saved_tensors

        x: input with arbitrary shape
        '''
        # Please think if we use "y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))", what might happen when x has a large absolute value
        # y = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

        # here we directly use torch.tanh(x) to avoid the problem above
        y = torch.tanh(x)

        # save an variable in ctx
        ctx.save_for_backward(y)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        grad_output: dL/dy
        grad_input: dL/dx = dL/dy * dy/dx, where y = forward(x)
        """
        # get an variable from ctx
        y, = ctx.saved_tensors

        # chain rule: dL/dx = dL/dy * dy/dx
        # where dL/dy = grad_output, and the dy/dx of tanh function is (1-y^2)!
        grad_input = grad_output * (1 - y ** 2)

        return grad_input

#TODO 1: complete the forward and backward functions of the Sigmoid activation function.
#Note: You can refer to the activation function Tanh
class Sigmoid(torch.autograd.Function):
    '''
    Sigmoid activation function
    y = 1 / (1 + exp(-x))
    '''

    @staticmethod
    def forward(ctx, x):

        # hint: you can use torch.exp(x) to calculate exp(x)
        ???

        # here we save y in ctx, in this way we can use y to calculate gradients in backward process
        ???

        return y

    @staticmethod
    def backward(ctx, grad_output):

        # get y from ctx
        ???

        # implement gradient of x (grad_input), grad_input refers to dL/dx
        # chain rule: dL/dx = dL/dy * dy/dx
        # where dL/dy = grad_output, and dy/dx of Sigmoid function is y * (1 - y)
        ???

        return grad_input

#TODO 2: complete the forward and backward functions of the ReLU activation function.
#Note: You can refer to the activation function Tanh
class ReLU(torch.autograd.Function):
    '''
    ReLU activation function
    y = max{x, 0}
    '''

    @staticmethod
    def forward(ctx, x):

        # set elements less than 0 in x to 0
        # this operation is inplace
        ???

        # save x in ctx, in this way we can use x to calculate gradients in backward process
        ???

        # return the output
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        # get x from ctx
        ???

        # chain rule: dL/dx = dL/dy * dy/dx
        # where dL/dy = grad_output, and dy/dx of ReLU function is 1 if x > 0, and 0 if x <= 0
        ???
        
        return grad_input


# activate function class according to the type
class Activation(nn.Module):
    def __init__(self, type):
        '''
        :param type:  'sigmoid', 'tanh', or 'relu'
        '''
        super().__init__()

        if type == 'sigmoid':
            self.act = Sigmoid.apply
        elif type == 'tanh':
            self.act = Tanh.apply
        elif type == 'relu':
            self.act = ReLU.apply
        else:
            print('activation type should be one of [sigmoid, tanh, relu]')
            raise NotImplementedError

    def forward(self, x):
        return self.act(x)
