import torch


class RadarImageAnnotation(object):
    def __init__(self, h, w, kernel_size, rate):
        self.rate = rate
        self.h = h
        self.w = w
        self.pool = torch.nn.MaxPool2d(kernel_size, stride=1, padding=int(kernel_size / 2))

    def __call__(self, pic: torch.Tensor):
        rain_rate = torch.sum(pic > self.rate) / (self.h * self.w)
        image_p = self.pool(pic)
        image_eq = (image_p == pic) * (image_p != 0)
        n_centers = torch.sum(image_eq)
        return pic, torch.stack([rain_rate, n_centers])


class MaxNormalization(object):

    def __call__(self, pic: torch.Tensor):
        pic = pic.float()
        return pic / pic.max()
