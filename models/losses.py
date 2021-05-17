# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

import torch

torch.manual_seed(1)


def model_loss_single(output, target, masks):
    single_loss = masks * (output - target) ** 2
    loss = torch.mean(torch.sum(single_loss, dim=0) / torch.sum(masks, dim=0))

    return loss


def single_losses(model):
    return model.masks * (
            model(model.X).view(-1, model.OUTPUT_SIZE) - model.y) ** 2


def model_loss(output, target, masks):
    single_loss = torch.nn.functional.mse_loss(output, target,
                                                       reduction='none')
    single_loss = masks * single_loss
    loss = torch.sum(
        torch.sum(single_loss, dim=1) / torch.sum(torch.sum(masks, dim=1)))

    return loss


def quantile_loss(output, target, masks, q):
    single_loss = masks * ((output - target) * (output >= target) * q + (
            target - output) * (output < target) * (1 - q))
    loss = torch.mean(torch.sum(single_loss, dim=1) / torch.sum(masks, dim=1))

    return loss
