#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PGD algorithm for adversarial attacks.

This module implements the Projected Gradient Descent (PGD) algorithm for generating
adversarial examples. It includes the PGDSampler class, which can be used to create
adversarial samples by optimizing the input data based on a specified loss function.

References: https://github.com/cleverhans-lab/cleverhans/blob/master/cleverhans/torch/attacks/projected_gradient_descent.py
"""
import time
import torch
import numpy as np

from cleverhans.torch.utils import clip_eta
from cleverhans.torch.utils import optimize_linear


class PGDSampler(object):
    """Projected Gradient Descent (PGD) adversarial sampler.

    Implements the Basic Iterative Method (Kurakin et al. 2016) or
    Madry et al. (2017) method for adversarial sample generation.

    References:
        Kurakin et al. 2016: https://arxiv.org/pdf/1607.02533.pdf
        Madry et al. 2017: https://arxiv.org/pdf/1706.06083.pdf
    """

    def __init__(
        self,
        loss_function,
        eps=0.125,
        eps_iter=0.01,
        nb_iter=40,
        norm=np.inf,
        clip_min=None,
        clip_max=None,
        rand_init=True,
        rand_minmax=None,
    ):
        """Initializes PGDSampler.

        Args:
            loss_function (callable): Loss function for adversarial optimization.
            eps (float): Maximum perturbation.
            eps_iter (float): Step size for each attack iteration.
            nb_iter (int): Number of attack iterations.
            norm (float): Order of the norm (np.inf or 2).
            clip_min (float, optional): Minimum value for adversarial example.
            clip_max (float, optional): Maximum value for adversarial example.
            rand_init (bool): Whether to start from a randomly perturbed input.
            rand_minmax (float, optional): Range for random initialization.
        """
        self.loss_function = loss_function
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.norm = norm
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.rand_init = rand_init
        self.rand_minmax = rand_minmax

        if self.norm not in [np.inf, 2]:
            raise ValueError("Norm order must be either np.inf or 2.")
        if self.eps < 0:
            raise ValueError("eps must be greater than or equal to 0, got {} instead".format(self.eps))
        if self.eps_iter < 0:
            raise ValueError("eps_iter must be greater than or equal to 0, got {} instead".format(self.eps_iter))
        if self.nb_iter <= 0:
            raise ValueError("nb_iter must be greater than 0, got {} instead".format(self.nb_iter))

        assert self.eps_iter <= self.eps, (self.eps_iter, self.eps)
        if self.clip_min is not None and self.clip_max is not None:
            if self.clip_min > self.clip_max:
                raise ValueError(
                    "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                        self.clip_min, self.clip_max
                    )
                )
        if self.rand_minmax is None:
            self.rand_minmax = self.eps

    def clip_x(self, x):
        """Clips the input tensor to the specified min and max values.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Clipped tensor.
        """
        if self.clip_min is not None or self.clip_max is not None:
            x = torch.clamp(x, self.clip_min, self.clip_max)
        return x

    def __call__(self, x, **kwargs):
        """Generates adversarial samples using PGD.

        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Additional arguments for the loss function.

        Returns:
            dict: Dictionary with adversarial sample, perturbation, and loss.
        """
        if self.eps == 0 or self.eps_iter == 0:
            return x
        if self.clip_min is not None:
            assert torch.all(torch.ge(x, torch.tensor(self.clip_min, device=x.device, dtype=x.dtype)))
        if self.clip_max is not None:
            assert torch.all(torch.le(x, torch.tensor(self.clip_max, device=x.device, dtype=x.dtype)))
        if self.rand_init:
            eta = torch.zeros_like(x).uniform_(-self.rand_minmax, self.rand_minmax)
            eta = clip_eta(eta, self.norm, self.eps)
            adv_x = self.clip_x(x + eta)
        else:
            adv_x = x.clone()
        ltm = time.time()
        for i in range(self.nb_iter):
            adv_x = adv_x.detach().requires_grad_(True)
            loss = self.loss_function(adv_x, **kwargs)
            loss.backward()
            perturbation = optimize_linear(adv_x.grad, self.eps_iter, self.norm)
            adv_x = self.clip_x(adv_x + perturbation)
            eta = clip_eta(adv_x - x, self.norm, self.eps)
            adv_x = self.clip_x(x + eta)
            if time.time() - ltm >= 5:
                print(
                    f"{i+1:4d}, Loss: {loss.item():6f}, "
                    f"ETA Mean: {eta.mean().item():6f}, ETA Std: {eta.std().item():6f}, "
                    f"ADV Mean: {adv_x.mean().item():6f}, ADV Std: {adv_x.std().item():6f}"
                )
                ltm = time.time()
        print(
            f"{i+1:4d}, Loss: {loss.item():6f}, "
            f"ETA Mean: {eta.mean().item():6f}, ETA Std: {eta.std().item():6f}, "
            f"ADV Mean: {adv_x.mean().item():6f}, ADV Std: {adv_x.std().item():6f}"
        )
        return {"pixel_values": adv_x, "perturbation": eta, "loss": loss}
