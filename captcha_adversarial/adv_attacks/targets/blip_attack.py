#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 viewstar000
"""
import torch
import numpy as np

from PIL import Image
from transformers import AutoProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration
from ..algorithms.bert_attack import BertAttacker, BasePredictorFactory
from ..algorithms.bsa_attack import BSALoss, BSAttacker, pixel_values_to_image
from ..algorithms.pgd_attack import PGDSampler
from ..algorithms.vla_attack import VLAttacker


class BlipVQAPredictorFactory(BasePredictorFactory):
    """Factory for BLIP VQA predictor."""

    def __init__(self, max_length=512, model_id="Salesforce/blip-vqa-base"):
        """Initialize BlipVQAPredictorFactory.

        Args:
            max_length (int): Maximum sequence length.
            model_id (str): Huggingface model identifier.
        """
        print(f"Init BlipVQAPredictorFactory ...")
        self.model_id = model_id
        self.max_length = max_length
        self.model = BlipForQuestionAnswering.from_pretrained(self.model_id, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(self.model_id, device_map="auto", use_fast=True)
        self.model.eval()
        self.device = self.model.device
        self.prompt_tpl = "{question} Please answer yes or no: "
        self.mask_token = self.processor.tokenizer.unk_token
        self.yes_token_id = self.processor.tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = self.processor.tokenizer.convert_tokens_to_ids("no")
        self.sep_token_id = self.processor.tokenizer.sep_token_id

    def predict(self, seq, image):
        """Predict the label and probability for a text-image pair.

        Args:
            seq (str): Input text.
            image (str or PIL.Image): Input image.

        Returns:
            Tuple[Tensor, Tensor]: (Label, probability tensor).
        """
        if isinstance(image, str):
            image = Image.open(image)
        inputs = self.processor(
            text=self.prompt_tpl.format(question=seq),
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            generated_outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                output_logits=True,
                return_dict_in_generate=True,
            )
            result = self.processor.batch_decode(generated_outputs.sequences, skip_special_tokens=True)[0]
            label = torch.Tensor([0 if result == "yes" else 1]).squeeze().type(torch.int64).to(self.device)
            probs = [0.0, 0.0]
            if len(generated_outputs.logits) >= 2:
                generated_probs = [
                    torch.softmax(generated_outputs.logits[0], dim=-1).squeeze(),
                    torch.softmax(generated_outputs.logits[1], dim=-1).squeeze(),
                ]
                probs[0] = generated_probs[0][self.yes_token_id].item() * generated_probs[1][self.sep_token_id].item()
                probs[1] = generated_probs[0][self.no_token_id].item() * generated_probs[1][self.sep_token_id].item()
            probs = torch.Tensor(probs).to(self.device)
            return label, probs


class BlipBertAttacker(BertAttacker):
    """BERT-based attacker for BLIP VQA."""

    def __init__(self, predictor_factory=None, **kwargs):
        """Initialize BlipBertAttacker.

        Args:
            predictor_factory: Predictor factory.
            **kwargs: Additional arguments.
        """
        print(f"Init BlipBertAttacker ...")
        predictor_factory = predictor_factory or BlipVQAPredictorFactory()
        super().__init__(predictor_factory, **kwargs)


class BlipModelBSALoss(BSALoss):
    """BSA loss for BLIP models."""

    def __init__(self, model=None, model_id="Salesforce/blip-image-captioning-base"):
        """Initializes BlipModelBSALoss.

        Args:
            model: Model instance.
            model_id (str): Model identifier.
        """
        print("Init BlipModelBSALoss ...")
        model = model or BlipForConditionalGeneration.from_pretrained(model_id, device_map="auto")
        super().__init__(model=model, model_id=model_id)


class BlipBSAttacker(BSAttacker):
    """Adversarial attacker for BLIP models using BSA loss."""

    def __init__(self, loss=None, sampler=None, **kwargs):
        """Initializes BlipBSAttacker.

        Args:
            processor: Preprocessing processor.
            loss: Loss function instance.
            sampler: Sampler instance.
            **kwargs: Additional arguments for sampler.
        """
        print("Init BlipBSAttacker ...")
        super().__init__(loss=loss or BlipModelBSALoss(), sampler=sampler, **kwargs)
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
        self.image_mean = self.processor.image_processor.image_mean
        self.image_std = self.processor.image_processor.image_std

    def pre_process(self, image, text, **kwargs):
        image, text, kwargs = super().pre_process(image, text, **kwargs)
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a PIL Image or a numpy array")
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True, **kwargs).to(self.device)
        return inputs.pixel_values.clone(), text, inputs


class BlipVLAttacker(VLAttacker):
    """Visual-Linguistic Adversarial Attacker for BLIP models."""

    def __init__(
        self,
        predictor_factory=None,
        first_bsa_attacker=None,
        iter_bsa_attacker=None,
        bert_attacker=None,
        bsa_loss=None,
    ):
        """Initializes BlipVLAttacker.

        Args:
            predictor_factory: Factory for creating BLIP predictor instances.
            first_bsa_attacker: Initial image attacker for BLIP.
            iter_bsa_attacker: Iterative image attacker for BLIP.
            bert_attacker: Text attacker for BLIP.
            bsa_loss: Loss function for BSA attacks.
        """
        print("Init BlipVLAttacker ...")
        predictor_factory = predictor_factory or BlipVQAPredictorFactory()
        bsa_loss = bsa_loss or BlipModelBSALoss()
        first_bsa_attacker = first_bsa_attacker or BlipBSAttacker(loss=bsa_loss, nb_iter=50)
        iter_bsa_attacker = iter_bsa_attacker or BlipBSAttacker(loss=bsa_loss, nb_iter=5)
        bert_attacker = bert_attacker or BlipBertAttacker(predictor_factory=predictor_factory)
        super().__init__(predictor_factory, first_bsa_attacker, iter_bsa_attacker, bert_attacker)
