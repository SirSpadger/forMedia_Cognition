#========================================================
#             Media and Cognition
#             Homework 1 Neural network basics
#             losses.py - loss functions
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2024
#========================================================

import torch
import torch.nn.functional as F

'''
In this script we will implement our MSE and Cross Entropy loss functions, including both the forward and backward processes.
More details about customizing a backward process can be found in:
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
'''

# here is the sample code of MSELoss
# you can use this as reference to implement the CrossEntropyLoss
class MSELoss(torch.autograd.Function):
    '''
    MSE loss function
    loss = (label - pred) ** 2
    '''

    @staticmethod
    def forward(ctx, pred, label):
        """
        :param pred: prediction with shape [batch_size, *], where ∗ means additional dimensions
        :param label: groundtruth, same shape as the predition
        :return: MSE loss, averaged by batch_size
        """

        # step 1: here we compute the summation of loss for each element and save both pred and label in ctx
        loss = torch.sum((pred - label) ** 2)
        ctx.save_for_backward(pred, label)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param grad_output: for loss function, grad_output will be 1
        """

        # step 2: get pred and label from ctx and calculate the derivative of loss w.r.t. pred (dL/dpred)
        pred, label = ctx.saved_tensors
        grad_input = grad_output * 2 * (pred - label)

        # return None for gradient of label since we do not need to compute dL/dlabel
        return grad_input, None

#TODO 1: Complete the CrossEntropyLoss loss function
class CrossEntropyLoss(torch.autograd.Function):
    '''
    Cross entropy loss function:
        loss = - log q_i
    where
        q_i = softmax(z_i) = exp(z_i) / (exp(z_0) + exp(z_1) + ...)

    However, when z_i has a lager value, exp(z_i) might become infinity.
    So we use stable softmax:
        softmax(z_i) = A exp(z_i) / A (exp(z_0) + exp(z_1) + ...)
    where
        A = exp(-z_max) = exp(-max{z_0, z_1, ...})
    therefore we have
        softmax(z_i) = softmax(z_i - z_max)
    '''

    @staticmethod
    def forward(ctx, logits, label):
        """
        :param logits: logits with shape [batch_size, n_classes], denoted by "z" in the above formula
        :param label: groundtruth with shape [batch_size], where 0 <= label[i] < n_classes - 1
        :return: cross entropy loss, averaged by batch_size
        """

        # step 1: calculate softmax(z) using stable softmax method
        # hint: you can use torch.exp(x) to calculate exp(x), and remember to convert label into one-hot version
        #e.g., if label = [0, 2] and n_classes=4, then the one-hot version is [[1,0,0,0], [0,0,1,0]]

        # calculate z_max
        ???

        # calculate exps = exp(z - z_max)
        ???

        # calculate q = softmax(y - y_max)
        ???

        # step 2: convert label into one-hot version
        # e.g., if label = [0, 2] and n_classes=4, then the one-hot version is [[1,0,0,0], [0,0,1,0]] 
        # the converted label has shape [batch_size, n_classes]
        # tips: you can use torch.nn.functional.one_hot() to convert label into one-hot vector with dimension n_classes
        ???

        # step 3: calculate cross entropy loss = - log q_i, and averaged by batch
        # save result of softmax and one-hot label in ctx for gradient computation
        ???

        pass 

    @staticmethod
    def backward(ctx, grad_output):

        # step 4: get q and label from ctx and calculate the derivative of loss w.r.t. pred (dL/dz)
        ???

        # return the pred (dL/dz) and None for dL/dlabel since we do not need to compute dL/dlabel
        ???