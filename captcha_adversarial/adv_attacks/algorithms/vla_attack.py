#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visual-Language Adversarial Attack (VLA) module.

This module implements a multi-stage adversarial attack pipeline for vision-language models,
combining image and text attacks to generate adversarial examples.

References:
    "VLAttack: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models"
    by Ziyi Yin, Muchao Ye, Tianrong Zhang, Tianyu Du, Jinguo Zhu, Han Liu, Jinghui Chen, Ting Wang and Fenglong Ma.
    https://arxiv.org/abs/2310.04655

Copyright (c) 2025 viewstar000
"""

from .bsa_attack import BSAttacker
from .bert_attack import BertAttacker


class VLAttacker(object):
    """Visual-Language Adversarial Attacker.

    This class performs a multi-stage adversarial attack on vision-language models,
    combining image and text attacks to maximize attack success.
    """

    def __init__(self, predictor_factory=None, first_bsa_attacker=None, iter_bsa_attacker=None, bert_attacker=None):
        """Initializes VLAttacker.

        Args:
            predictor_factory: Factory for creating predictor instances.
            first_bsa_attacker: Attacker for the initial image attack stage.
            iter_bsa_attacker: Attacker for iterative image attack stages.
            bert_attacker: Attacker for text-based adversarial attack.
        """
        print("Init VLAttacker ...")
        self.predictor_factory = predictor_factory
        self.first_bsa_attacker = first_bsa_attacker or BSAttacker()
        self.iter_bsa_attacker = iter_bsa_attacker or first_bsa_attacker or BSAttacker()
        self.bert_attacker = bert_attacker or BertAttacker(predictor_factory=predictor_factory)

    def __call__(self, origin_image, origin_text, similarity_threshold=0.95, vla_iters=10):
        """Performs a visual-Language adversarial attack.

        Args:
            origin_image: The original image (PIL.Image).
            origin_text (str): The original text.
            similarity_threshold (float): Similarity threshold for text attack filtering.
            vla_iters (int): Maximum number of VLA iterations.

        Returns:
            dict: Dictionary containing attack results and adversarial examples.
        """
        origin_label, origin_probs = self.predictor_factory.predict(origin_text, origin_image)
        print(f"Original label: {origin_label.item()}, Original probs: {origin_probs}")
        adv_outputs = self.first_bsa_attacker(origin_image, origin_text)
        print(
            "Image pixel values: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(
                adv_outputs["pixel_values"].mean().item(),
                adv_outputs["pixel_values"].std().item(),
                adv_outputs["pixel_values"].min().item(),
                adv_outputs["pixel_values"].max().item(),
            )
        )
        adv_image = adv_outputs["adv_image"]
        label, probs = self.predictor_factory.predict(origin_text, adv_image)
        print(f"Adversarial label: {label.item()}, Adversarial probs: {probs}")
        if label != origin_label:
            print("Adversarial attack success!")
            adv_outputs["success"] = True
            adv_outputs["stage"] = "BSA"
            adv_outputs["text"] = origin_text
            adv_outputs["label"] = label
            adv_outputs["probs"] = probs
            return adv_outputs
        bert_outputs = self.bert_attacker(origin_text, origin_image, similarity_threshold=similarity_threshold)
        if bert_outputs["success"]:
            print("Bert attack success!")
            bert_outputs["adv_image"] = origin_image
            bert_outputs["stage"] = "BERT"
            return adv_outputs
        candidats = bert_outputs["candidates"][:vla_iters]
        for _, adv_text in candidats:
            print(f"Adversarial text: {adv_text}")
            adv_outputs = self.iter_bsa_attacker(adv_image, adv_text)
            print(
                "Image pixel values: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(
                    adv_outputs["pixel_values"].mean().item(),
                    adv_outputs["pixel_values"].std().item(),
                    adv_outputs["pixel_values"].min().item(),
                    adv_outputs["pixel_values"].max().item(),
                )
            )
            adv_image = adv_outputs["adv_image"]
            label, probs = self.predictor_factory.predict(adv_text, adv_image)
            print(f"Adversarial label: {label.item()}, Adversarial probs: {probs}")
            if label != origin_label:
                print("Adversarial attack success!")
                adv_outputs["success"] = True
                adv_outputs["stage"] = "VLA"
                adv_outputs["text"] = adv_text
                adv_outputs["label"] = label
                adv_outputs["probs"] = probs
                return adv_outputs
        print("Adversarial attack failed!")
        adv_outputs["success"] = False
        adv_outputs["stage"] = "VLA"
        adv_outputs["text"] = adv_text
        adv_outputs["label"] = label
        adv_outputs["probs"] = probs
        return adv_outputs
