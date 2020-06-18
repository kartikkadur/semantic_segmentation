# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 00:47:02 2020

@author: Karthik
"""
import torch
import argparse

from models import mynet
from datasets.dataloaders import get_voc_loader

from utils import AverageMeter

#mean iou calculation
OFFSET = 1e-6

def miou(outputs, labels):
    outputs = outputs.squeeze(1)
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))
    #Adding OFFSET to avoid divide by zero 
    iou = (intersection) / (union + OFFSET)
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    
    return thresholded.mean()

def train(net, train_loader, optim):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    train_loss = AverageMeter()
    torch.cuda.empty_cache()
    net.train()
    for i, data in enumerate(train_loader):
        inputs, gts = data
        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)

        inputs, gts = inputs.cuda(), gts.cuda()
        optim.zero_grad()

        loss = net(inputs, gts)
        loss = loss.clone().detach_()

        train_loss.update(loss.item(), batch_pixel_size)
        optim.step()
    return loss


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--root', metavar='N',
                        help='root dir of the dataset')
    parser.add_argument('--set', default="train", choices = ["train", "test"],
                        help='Which split to use. Train/Test')
    parser.add_argument("--year", default = '2007', help = "Dataset year")
    parser.add_argument("--download", type = bool, default = False,
                        help = "True if the dataset has to be downloaded, False otherwise.")
    parser.add_argument("--bs", type = int, default = 32, help = "Batch size")
    parser.add_argument("--lr", type = float, default = 0.01, help = "Learning rate")
    args = parser.parse_args()

    #criterion
    criterion = torch.nn.BCELoss()

    #check if cuda is available
    cuda = torch.cuda.is_available()
    activation = torch.nn.Sigmoid()

    #get dataloders
    train_loader = get_voc_loader(args.root, args.set, False, args.year
                                  , batch_size = args.bs)

    # create the model class
    model = mynet.SegNet(21, criterion)

    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)

    if cuda:
        model = model.cuda()
    
    loss = train(model, train_loader, optimizer)
    print(loss)