# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 00:05:20 2020

@author: Karthik
"""
try:
    from voc import PascalVOC
except:
    from .voc import PascalVOC

from torch.utils.data import DataLoader


def get_voc_loader(root,
                   image_set = 'train',
                   download = False,
                   year = '2012',
                   transform = None,
                   batch_size = 32,
                   shuffle = True,
                   drop_last = True
                  ):
    dataset = PascalVOC(root, image_set, download, year, transform)
    dataloader = DataLoader(dataset,
                            batch_size = batch_size,
                            shuffle=True,
                            drop_last=True
                           )
    return dataloader


if __name__ == "__main__":
    loader = get_voc_loader(r"C:\Users\Karthik\Downloads\DATA", download = True)