#!/usr/bin/env python

import torch

###############################################################################


def find_device(use_available_device: bool | str = True) -> str:
    if isinstance(use_available_device, str):
        return use_available_device
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"
