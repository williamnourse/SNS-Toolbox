"""
Various Utility functions used when creating/simulating the various backends.
"""

import torch


def send_vars(variables, device):
    """
    Send a set of variables to a specific device. Used with PyTorch backends.

    :param variables:   Variables to send.
    :type variables:    List of torch.tensors
    :param device:      Device to send the variables to.
    :type device:       torch.device
    :return:            Variables sent to correct device.
    :rtype:             List of torch.tensors
    """
    for var in variables:
        var = var.to(device)
    torch.cuda.empty_cache()
    return variables
