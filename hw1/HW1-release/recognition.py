#========================================================
#             Media and Cognition
#             Homework 1 Neural network basics
#             recognition.py - character classification
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2024
#========================================================

# ==== Part 0: import libs
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import json, cv2, os, string
import matplotlib.pyplot as plt

# this time we implement our networks and loss functions in other python script, and import them here
from network import MLP
from losses import CrossEntropyLoss

# argparse is used to conveniently set our configurations
import argparse

# ==== Part 1: data loader

# construct a dataset and a data loader, more details can be found in
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataloader

class ListDataset(Dataset):
    def __init__(self, im_dir, file_path, norm_size=(32, 32)):
        '''
        :param im_dir: path to directory with images
        :param file_path: json file containing image names and labels
        :param norm_size: image normalization size, (height, width)
        '''

        # this time we will try to recognize 26 English letters (case-insensitive)
        letters = string.ascii_letters[-26:]  # ABCD...XYZ
        self.alphabet = {letters[i]:i for i in range(len(letters))}
        self.norm_size = norm_size

        with open(file_path, 'r') as f:
            imgs = json.load(f)
            im_names = list(imgs.keys())

            self.im_paths = [os.path.join(im_dir, im_name) for im_name in im_names]
            self.labels = list(imgs.values())

    def __len__(self):
        # the __len__() function should return the total number of samples in the dataset
        return len(self.im_paths)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # read an image and convert it to grey scale
        im_path = self.im_paths[index]
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # image pre-processing, after pre-processing, the size of the image should be as norm_size and the values of image pixels should be within [-1,1]
        im = cv2.resize(im, self.norm_size)
        im = im / 255.
        im = (im - 0.5) * 2.0

        # get the label of the current image
        # upper() is used to convert a letter into uppercase
        label = self.labels[index].upper()

        # convert an English letter into a number index
        label = self.alphabet[label]

        # TODO 1: return the image and its label
        ???
        


def dataLoader(im_dir, file_path, norm_size, batch_size, workers=0):
    '''
    :param im_dir: path to directory with images
    :param file_path: file with image paths and labels
    :param norm_size: image normalization size, (height, width)
    :param batch_size: batch size
    :param workers: number of workers for loading data in multiple threads
    :return: a data loader
    '''

    dataset = ListDataset(im_dir, file_path, norm_size)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True if 'train' in file_path else False,  # shuffle images only when training
                      num_workers=workers)


# ==== Part 2: training, validation and testing

def train_val(model, trainloader, valloader, n_epochs, 
              lr, optim_type, momentum, weight_decay,
              valInterval, device='cpu'):
    '''
    The main training procedure
    ----------------------------
    :param model: the MLP model
    :param trainloader: the dataloader of the train set
    :param valloader: the dataloader of the validation set
    :param n_epochs: number of training epochs
    :param lr: learning rate
    :param optim_type: optimizer, can be 'sgd', 'adagrad', 'rmsprop', 'adam', or 'adadelta'
    :param momentum: only used if optim_type == 'sgd'
    :param weight_decay: the factor of L2 penalty on network weights
    :param valInterval: the frequency of validation, e.g., if valInterval = 5, then do validation after each 5 training epochs
    :param device: 'cpu' or 'cuda', we can use 'cpu' for our homework if GPU with cuda support is not available
    '''

    # define the cross entropy loss function.
    ce_loss = CrossEntropyLoss.apply

    # optimizer
    if optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr, weight_decay=weight_decay)
    elif optim_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr, weight_decay=weight_decay)
    elif optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    elif optim_type == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr, weight_decay=weight_decay)
    else:
        print('[Error] optim_type should be one of sgd, adagrad, rmsprop, adam, or adadelta')
        raise NotImplementedError

    # training

    # to save loss of each training epoch in a python "list" data structure
    losses = []

    for epoch in range(n_epochs):
        # set the model in training mode
        model.train()

        # to save total loss in one epoch
        total_loss = 0.

        #TODO 2: Calculate losses and train the network using the optimizer
        for ??? :  # get a batch of data

            # step 1: set data type and device
            ???

            # step 2: convert an image to a vector as the input of the MLP
            ???

            # hit: clear gradients in the optimizer
            ???

            # step 3: run the model which is the forward process
            ???

            # step 4: compute the loss, and call backward propagation function
            ???

            # step 5: sum up of total loss, loss.item() return the value of the tensor as a standard python number
            # this operation is not differentiable
            ???

            # step 6: call a function, optimizer.step(), to update the parameters of the models
            ???

        # average of the total loss for iterations
        avg_loss = total_loss / len(trainloader)
        losses.append(avg_loss)
        print('Epoch {:02d}: loss = {:.3f}'.format(epoch + 1, avg_loss))

        # validation
        if (epoch + 1) % valInterval == 0:
            val_acc = test(model, valloader, device)
            # show prediction accuracy
            print('Epoch {:02d}: validation accuracy = {:.1f}%'.format(epoch + 1, 100 * val_acc))


    # save model parameters in a file
    model_save_path = 'saved_models/recognition.pth'.format(epoch + 1)

    torch.save({'state_dict': model.state_dict(),
                }, model_save_path)
    print('Model saved in {}\n'.format(model_save_path))

    # draw the loss curve
    plot_loss(losses)


def test(model, testloader, device):
    '''
    The testing procedure
    ----------------------------
    :param model: the MLP model
    :param testloader: the dataloader to be tested/validated
    :param device: 'cpu' or 'cuda', we can use 'cpu' for our homework if GPU with cuda support is not available
    '''
    # set the model in evaluation mode
    model.eval()

    n_correct = 0.  # number of images that are correctly classified
    n_imgs = 0.  # number of total images
    
    with torch.no_grad():  # we do not need to compute gradients during validation

        #TODO 3: get the prediction of the data and calculate the accuracy
        for imgs, labels in testloader:
            # step 1: set data type and device
            ???

            # step 2: convert an image to a vector as the input of the MLP
            ???

            # step 3: run the model which is the forward process
            ???

            # step 4: get the predicted value by the output using out.argmax(1)
            ???

            # step 5: sum up the number of images correctly recognized and the total image number
            ???
    accuracy = n_correct / n_imgs
    return accuracy


# ==== Part 3: predict new images
def predict(model, im_path, norm_size, device):
    '''
    The predicting procedure
    ---------------
    :param model: the MLP model
    :param im_path: path of an image
    :param norm_size: image normalization size, (height, width)
    :param device: 'cpu' or 'cuda', we can use 'cpu' for our homework if GPU with cuda support is not available
    '''

    # TODO 4: enter the evaluation mode
    ???

    # TODO 4: image pre-processing, similar to what we do in ListDataset()
    ???

    # convert im from numpy.ndarray to torch.tensor
    im = torch.from_numpy(im)

    # input im into the model
    with torch.no_grad():
        input = im.view(1, -1).type(torch.float32).to(device)
        out = model(input)
        prediction = out.argmax(1)[0].item()

    # convert index of prediction to the corresponding character
    letters = string.ascii_letters[-26:]  # ABCD...XYZ
    prediction = letters[prediction]

    print('Prediction: {}'.format(prediction))


# ==== Part 4: draw the loss curve
def plot_loss(losses):
    '''
    :param losses: list of losses for each epoch
    :return:
    '''

    f, ax = plt.subplots()

    # draw loss
    ax.plot(losses)

    # set labels
    ax.set_xlabel('training epoch')
    ax.set_ylabel('loss')

    # show the plots
    plt.show()


if __name__ == '__main__':
    # set random seed for reproducibility
    seed = 2023
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # set configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train, test or predict')
    parser.add_argument('--im_dir', type=str, default='data/character_classification/images',
                        help='path to directory with images')
    parser.add_argument('--train_file_path', type=str, default='data/character_classification/train.json',
                        help='file list of training image paths and labels')
    parser.add_argument('--val_file_path', type=str, default='data/character_classification/validation.json',
                        help='file list of validation image paths and labels')
    parser.add_argument('--test_file_path', type=str, default='data/character_classification/test.json',
                        help='file list of test image paths and labels')
    parser.add_argument('--batchsize', type=int, default=8, help='batch size')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')

    # configurations for training
    parser.add_argument('--hsize', type=str, default='32', help='hidden size for each hidden layer, splitted by comma')
    parser.add_argument('--layer', type=int, default=2, help='number of layers in the MLP')
    parser.add_argument('--act', type=str, default='relu',
                        help='type of activation function, can be sigmoid, tanh, or relu')
    parser.add_argument('--norm_size', type=tuple, default=(32, 32), help='image normalization size, (height, width)')
    parser.add_argument('--epoch', type=int, default=50, help='number of training epochs')
    parser.add_argument('--n_classes', type=int, default=26, help='number of classes')
    parser.add_argument('--valInterval', type=int, default=10, help='the frequency of validation')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--optim_type', type=str, default='sgd', help='type of optimizer, can be sgd, adagrad, rmsprop, adam, or adadelta')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of the SGD optimizer, only used if optim_type is sgd')
    parser.add_argument('--weight_decay', type=float, default=0., help='the factor of L2 penalty on network weights')

    # configurations for test and prediction
    parser.add_argument('--model_path', type=str, default='saved_models/recognition.pth', help='path of a saved model')
    parser.add_argument('--im_path', type=str, default='data/character_classification/new_images/predict01.png',
                        help='path of an image to be recognized')

    opt = parser.parse_args()

    # TODO 5: initialize the MLP model
    # what is the input size of the MLP?
    # hint 1: we convert an image to a vector as the input of the MLP
    # hint 2: each image has shape [norm_size[0], norm_size[1]]
    model = ???

    # for the 'test' and 'predict' mode, we should load the saved checkpoint into the model
    if opt.mode == 'test' or opt.mode == 'predict':
        checkpoint = torch.load(opt.model_path, map_location='cpu')
        # load model parameters we saved in model_path
        model.load_state_dict(checkpoint['state_dict'])
        print('[Info] Load model from {}'.format(opt.model_path))

    # put the model on CPU or GPU according to the device in args
    model = ???

    # -- run the code for training and validation
    if opt.mode == 'train':
        # training and validation data loader
        trainloader = dataLoader(opt.im_dir, opt.train_file_path, opt.norm_size, opt.batchsize)
        valloader = dataLoader(opt.im_dir, opt.val_file_path, opt.norm_size, opt.batchsize)
        train_val(model, trainloader, valloader,
                  n_epochs=opt.epoch,
                  lr=opt.lr,
                  optim_type=opt.optim_type,
                  momentum=opt.momentum,
                  weight_decay=opt.weight_decay,
                  valInterval=opt.valInterval,
                  device=opt.device)

    # -- test the saved model
    elif opt.mode == 'test':
        testloader = dataLoader(opt.im_dir, opt.test_file_path, opt.norm_size, opt.batchsize)
        acc = test(model, testloader, opt.device)
        print('[Info] Test accuracy = {:.1f}%'.format(100 * acc))

    # -- predict a new image
    elif opt.mode == 'predict':
        predict(model, im_path=opt.im_path, norm_size=opt.norm_size, device=opt.device)

    else:
        print('mode should be train, test, or predict')
        raise NotImplementedError
