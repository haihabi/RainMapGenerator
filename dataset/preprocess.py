import torch


class MaxNormalization(object):

    def __call__(self, pic: torch.Tensor):
        pic = pic.float()
        return pic / pic.max()
