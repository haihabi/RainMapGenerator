import torch


class MaxNormalization(object):

    def __call__(self, pic: torch.Tensor):
        return pic / pic.max()
