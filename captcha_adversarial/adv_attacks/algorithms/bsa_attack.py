#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BSA Algorithm for adversarial attacks.

References:
    "VLAttack: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models"
    by Ziyi Yin, Muchao Ye, Tianrong Zhang, Tianyu Du, Jinguo Zhu, Han Liu, Jinghui Chen, Ting Wang and Fenglong Ma.
    https://arxiv.org/abs/2310.04655

Copyright (c) 2025 viewstar000
"""

import torch
import numpy as np

from PIL import Image
from .pgd_attack import PGDSampler


def display_hidden_states(val, level=0):
    """Recursively display the structure of the hidden states.

    Args:
        val: The value to display (can be tuple, dict, or tensor).
        level (int): Indentation level for pretty printing.
    """
    if isinstance(val, tuple):
        print("  " * level, "Tuple({})".format(len(val)))
        for v in val:
            display_hidden_states(v, level + 1)
    elif isinstance(val, dict):
        print("  " * level, "Dict({})".format(len(val)))
        for k, v in val.items():
            print("  " * (level + 1), k)
            display_hidden_states(v, level + 2)
    elif isinstance(val, torch.Tensor):
        if val.dtype in (torch.float32, torch.float16):
            print("  " * level, "Tensor({}, {})".format(val.dtype, val.shape), val.mean())
        else:
            print("  " * level, "Tensor({}, {})".format(val.dtype, val.shape))
    else:
        print("  " * level, type(val))


def pixel_values_to_image(pixel_values, ret_type="PIL", image_mean=None, image_std=None):
    """Convert pixel values (PyTorch tensor) to a PIL image or NumPy array.

    Args:
        pixel_values (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].
        ret_type (str): Return type, either "PIL" or "numpy".
        image_mean (list, optional): Mean values for denormalization.
        image_std (list, optional): Standard deviation values for denormalization.

    Returns:
        PIL.Image or numpy.ndarray: Converted image.
    """
    single_image = pixel_values.squeeze(0).detach().cpu()
    image_mean = [0.485, 0.456, 0.406] if image_mean is None else image_mean
    image_std = [0.229, 0.224, 0.225] if image_std is None else image_std
    denormalized_image = single_image * torch.tensor(image_std).view(3, 1, 1) + torch.tensor(image_mean).view(3, 1, 1)
    image_np = denormalized_image.detach().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    if ret_type == "PIL":
        image_pil = Image.fromarray(image_np)
        return image_pil
    else:
        return image_np


class BSALoss(object):
    """BSA loss for adversarial image generation."""

    def __init__(self, model=None, model_id=None):
        """Initializes BSALoss.

        Args:
            model: Model instance.
            model_id (str, optional): Model identifier.
        """
        self.model_id = model_id
        self.model = model
        if self.model is not None:
            self.model.eval()
            self.model.requires_grad_(False)

    def __call__(self, adv_image=None, ori_image=None, **kwargs):
        """Computes the BSA loss between adversarial and original images.

        Args:
            adv_image (torch.Tensor): Adversarial image tensor.
            ori_image (torch.Tensor): Original image tensor.
            **kwargs: Additional arguments for the model.

        Returns:
            torch.Tensor: Loss value.
        """
        if "pixel_values" in kwargs:
            if ori_image is None:
                ori_image = kwargs["pixel_values"]
            del kwargs["pixel_values"]
        adv_outputs = self.model(pixel_values=adv_image, return_dict=True, output_hidden_states=True, **kwargs)
        ori_outputs = self.model(pixel_values=ori_image, return_dict=True, output_hidden_states=True, **kwargs)
        adv_hidden_states = torch.cat(adv_outputs.hidden_states)
        ori_hidden_states = torch.cat(ori_outputs.hidden_states)
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        loss = -cos(adv_hidden_states, ori_hidden_states).sum()
        return loss


class BSAttacker(object):
    """Adversarial attacker for image-text models using BSA loss."""

    def __init__(self, loss=None, sampler=None, **kwargs):
        """Initializes BSAttacker.

        Args:
            processor: Preprocessing processor.
            loss: Loss function instance.
            sampler: Sampler instance.
            **kwargs: Additional arguments for sampler.
        """
        self.loss = loss or BSALoss()
        self.sampler = sampler or PGDSampler(self.loss, **kwargs)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.image_mean = None
        self.image_std = None

    def pre_process(self, image, text, **kwargs):
        return image, text, kwargs

    def post_process(self, outputs, inputs):

        outputs["adv_image"] = pixel_values_to_image(
            outputs["pixel_values"],
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
        outputs["ptb_image"] = pixel_values_to_image(
            outputs["perturbation"],
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
        outputs["loss"] = outputs["loss"].item()
        return outputs

    def __call__(self, image, text, **kwargs):
        """Performs adversarial attack on an image-text pair.

        Args:
            image (str, np.ndarray, or PIL.Image): Input image.
            text (str): Input text.

        Returns:
            dict: Dictionary with adversarial image, perturbation, and loss.
        """
        image, _, inputs = self.pre_process(image, text, **kwargs)
        outputs = self.sampler(image, **inputs)
        outputs = self.post_process(outputs, inputs)
        return outputs
