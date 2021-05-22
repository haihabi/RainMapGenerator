import torch
import numpy as np
from tqdm import tqdm
from metric.inception_v3 import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg


class FrechetInceptionDistance(object):
    def __init__(self, batch_size, ds_loader, input_working_device, z_dim=128, dim=64, conditional=False):
        with torch.no_grad():
            self.model = InceptionV3(output_blocks=[InceptionV3.BLOCK_INDEX_BY_DIM[dim]]).to(input_working_device)
        self.batch_size = batch_size
        self.conditional = conditional
        self.ds_loader = ds_loader
        self.ref_n_samples = len(ds_loader)
        ref_pred = self.get_activations(ds_loader)
        self.ref_mu, self.ref_sigma = self.calculate_activation_statistics(ref_pred)
        self.working_device = input_working_device

        print("Finish Ref Loading")

    def _get_pred(self, image):
        pred = self.model(image)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1)).cpu().detach().numpy()
        return pred

    def get_activations(self, ds_loader):
        pred_list = []
        for data in tqdm(ds_loader):
            if self.conditional:
                image = data[0]
            else:
                image = data

            image = image.float()
            image = image.repeat((1, 3, 1, 1)).cuda()
            pred_list.append(self._get_pred(image))
            # pred = model(image)[0]
            #
            # # If model output is not scalar, apply global spatial average pooling.
            # # This happens if you choose a dimensionality not equal 2048.
            # if pred.shape[2] != 1 or pred.shape[3] != 1:
            #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1)).cpu().numpy()
            # pred_list.append(pred)
        act = np.concatenate(pred_list, axis=0)
        return act.reshape(act.shape[0], -1)

    @staticmethod
    def calculate_activation_statistics(activation_array):
        mu = np.mean(activation_array, axis=0)
        sigma = np.cov(activation_array, rowvar=False)
        return mu, sigma

    def calculate_fid(self, generator):
        pred_list = []
        for data in tqdm(self.ds_loader):
            if self.conditional:
                label = data[1]
            else:
                label = None
            y = generator(self.batch_size, cond=label)
            if isinstance(y, tuple):
                y = y[0]
            if y.shape[1] == 1:
                y = y.repeat(1, 3, 1, 1)
            pred_list.append(self._get_pred(y))
        pred_gen = np.concatenate(pred_list, axis=0)
        pred_gen = pred_gen.reshape(pred_gen.shape[0], -1)
        mu, sigma = self.calculate_activation_statistics(pred_gen)
        return self.calculate_frechet_distance(mu, sigma, self.ref_mu, self.ref_sigma)

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
