"""
Utility functions used when creating/simulating the various backends
William Nourse
September 13, 2021
You called the guy 'Snot'?
"""

import torch


def sendVars(variables, device):
    for var in variables:
        var = var.to(device)
    torch.cuda.empty_cache()
    return variables