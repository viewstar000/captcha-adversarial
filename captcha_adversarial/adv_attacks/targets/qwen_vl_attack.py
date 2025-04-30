#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 viewstar000
"""
import torch
import weakref
import numpy as np

from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from ..algorithms.bsa_attack import BSALoss, BSAttacker
from ..algorithms.bert_attack import BasePredictorFactory, BertAttacker
from ..algorithms.vla_attack import VLAttacker


__loaded_model_id = None
__loaded_model_ref = None


def load_model(model_id="Qwen/Qwen2.5-VL-3B-Instruct", **kwargs):
    """Load the Qwen2.5-VL model and cache it for future use."""
    global __loaded_model_id, __loaded_model_ref
    if __loaded_model_id == model_id and __loaded_model_ref is not None and __loaded_model_ref() is not None:
        return __loaded_model_ref()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        __loaded_model_id = model_id
        __loaded_model_ref = weakref.ref(model)
        return model


class QwenVL25ModelBSALoss(BSALoss):
    """Adversarial BSA loss for the Qwen2.5-VL model."""

    def __init__(self, model=None, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
        """Initializes the adversarial loss instance.

        Args:
            model: Pre-initialized Qwen2_5_VLForConditionalGeneration (optional).
            model_id (str): HuggingFace model identifier if not provided explicitly.
        """
        print("Init QwenVL25ModelBSALoss ...")
        model = model or load_model(model_id, device_map="auto")
        super().__init__(model=model, model_id=model_id)


class QwenVL25BSAttacker(BSAttacker):
    """Adversarial attacker implementation for the Qwen2.5-VL model."""

    def __init__(self, loss=None, sampler=None, **kwargs):
        """Initializes the adversarial attack instance.

        Args:
            loss: BSALoss instance (default uses QwenVL25ModelBSALoss).
            sampler: PGD-based pixel value sampler.
            **kwargs: Additional parameters for the sampler configuration.
        """
        print("Init QwenVL25BSAttacker ...")
        super().__init__(loss=loss or QwenVL25ModelBSALoss(), sampler=sampler, **kwargs)
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            max_pixels=384 * 384,
            use_fast=True,
        )
        self.image_mean = self.processor.image_processor.image_mean
        self.image_std = self.processor.image_processor.image_std

    def pre_process(self, image, text, **kwargs):
        """Preprocesses input data into model-ready format.

        Args:
            image: Input PIL Image, numpy array or path string.
            text (str): Text prompt to process with the image.
            **kwargs: Additional processing parameters.

        Returns:
            Tuple containing processed pixel values, text and full inputs.
        """
        image, text, kwargs = super().pre_process(image, text, **kwargs)
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a PIL Image or a numpy array")
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        return inputs.pixel_values.clone(), text, inputs

    def convert_pixels_to_CWH(self, pixels, image_grid_thw):
        """Converts model-processed pixel values to CWH format.

        Args:
            pixels (torch.Tensor): Processed pixel tensor from model.
            image_grid_thw (tuple): Image grid dimensions (T,H,W).

        Returns:
            torch.Tensor: Pixel values in channel-first format.
        """
        grid_t, grid_h, grid_w = image_grid_thw
        spatial_merge_size = 2
        temporal_patch_size = self.processor.image_processor.temporal_patch_size
        patch_size = self.processor.image_processor.patch_size
        pixels = pixels.reshape(
            int(grid_t),  # temporals
            int(grid_h / spatial_merge_size),  # H_patchs / spatial_merge_size
            int(grid_w / spatial_merge_size),  # W_patchs / spatial_merge_size
            spatial_merge_size,  # spatial_merge_height_size
            spatial_merge_size,  # spatial_merge_width_size
            -1,  # channels
            temporal_patch_size,  # temporal_patch_size
            patch_size,  # patch_height_size
            patch_size,  # patch_width_size
        )
        # Convert to B, C, H, W format
        pixels = pixels.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(
            int(grid_t * temporal_patch_size),
            -1,
            int(grid_h * patch_size),
            int(grid_w * patch_size),
        )
        return pixels[0]

    def post_process(self, outputs, inputs):
        """Final processing to prepare output images.

        Args:
            outputs (dict): Attack outputs containing pixel values.
            inputs (BatchEncoding): Original model inputs for metadata

        Returns:
            dict: Processed outputs ready for visualization
        """
        image_grid_thw = inputs.image_grid_thw.squeeze().detach().cpu().numpy()
        outputs["pixel_values"] = self.convert_pixels_to_CWH(outputs["pixel_values"], image_grid_thw)
        outputs["perturbation"] = self.convert_pixels_to_CWH(outputs["perturbation"], image_grid_thw)
        return super().post_process(outputs, inputs)


class QwenVL25Classifier(BasePredictorFactory):
    def __init__(self, max_length=512, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
        print("Init QwenVL25Classifier...")
        super(QwenVL25Classifier, self).__init__()
        self.model_id = model_id
        self.max_length = max_length
        self.model = load_model(self.model_id, device_map="auto")
        self.model.eval()
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(self.model_id, device_map="auto", use_fast=True)
        self.device = self.model.device
        self.mask_token = "[UNK]"
        self.yes_token_id = self.processor.tokenizer.convert_tokens_to_ids("YES")
        self.no_token_id = self.processor.tokenizer.convert_tokens_to_ids("NO")
        self.eos_token_id = self.processor.tokenizer.eos_token_id

    def predict(self, seq, image):
        """Predict the label and probability for a text-image pair.

        Args:
            seq (str): Input text.
            image (str or PIL.Image): Input image.

        Returns:
            Tuple[Tensor, Tensor]: (Label, probability tensor).
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a PIL Image or a numpy array")
        inputs = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": seq},
                        {"type": "text", "text": "\nPlease answer YES or NO!"},
                    ],
                }
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            generated_outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                output_logits=True,
                return_dict_in_generate=True,
            )
            generated_length = len(generated_outputs.logits)
            result = self.processor.batch_decode(
                generated_outputs.sequences[:, -generated_length:], skip_special_tokens=True
            )[0]
            print(f'QwenVL25Classifier "{seq.strip()}" -> {result.strip()}')
            label = torch.Tensor([0 if result.upper() == "YES" else 1]).squeeze().type(torch.int64).to(self.device)
            probs = [0.0, 0.0]
            if len(generated_outputs.logits) >= 2:
                generated_probs = [
                    torch.softmax(generated_outputs.logits[0], dim=-1).squeeze(),
                    torch.softmax(generated_outputs.logits[1], dim=-1).squeeze(),
                ]
                probs[0] = generated_probs[0][self.yes_token_id].item() * generated_probs[1][self.eos_token_id].item()
                probs[1] = generated_probs[0][self.no_token_id].item() * generated_probs[1][self.eos_token_id].item()
            probs = torch.Tensor(probs).to(self.device)
            return label, probs


class QwenVL25BertAttacker(BertAttacker):
    """BERT-based attacker for QwenVL25."""

    def __init__(self, predictor_factory=None, **kwargs):
        """Initialize QwenVL25BertAttacker.

        Args:
            predictor_factory: Predictor factory.
            **kwargs: Additional arguments.
        """
        print(f"Init QwenVL25BertAttacker ...")
        predictor_factory = predictor_factory or QwenVL25Classifier()
        super().__init__(predictor_factory, **kwargs)


class QwenVL25VLAttacker(VLAttacker):
    """Visual-Linguistic Adversarial Attacker for BLIP models."""

    def __init__(
        self,
        predictor_factory=None,
        first_bsa_attacker=None,
        iter_bsa_attacker=None,
        bert_attacker=None,
        bsa_loss=None,
    ):
        """Initializes QwenVL25VLAttacker.

        Args:
            predictor_factory: Factory for creating BLIP predictor instances.
            first_bsa_attacker: Initial image attacker for BLIP.
            iter_bsa_attacker: Iterative image attacker for BLIP.
            bert_attacker: Text attacker for BLIP.
            bsa_loss: Loss function for BSA attacks.
        """
        print("Init QwenVL25VLAttacker ...")
        predictor_factory = predictor_factory or QwenVL25Classifier()
        bsa_loss = bsa_loss or QwenVL25ModelBSALoss()
        first_bsa_attacker = first_bsa_attacker or QwenVL25BSAttacker(loss=bsa_loss, nb_iter=50)
        iter_bsa_attacker = iter_bsa_attacker or QwenVL25BSAttacker(loss=bsa_loss, nb_iter=5)
        bert_attacker = bert_attacker or QwenVL25BertAttacker(predictor_factory=predictor_factory)
        super().__init__(predictor_factory, first_bsa_attacker, iter_bsa_attacker, bert_attacker)
