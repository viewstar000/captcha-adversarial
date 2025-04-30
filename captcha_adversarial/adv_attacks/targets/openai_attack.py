#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 viewstar000
"""
import base64
import torch

from io import BytesIO
from PIL import Image
from openai import OpenAI
from ..algorithms.bert_attack import BasePredictorFactory


class OpenAIMultiModalClassifier(BasePredictorFactory):

    def __init__(self, model_id=None, api_base_url=None, api_key=None):
        super().__init__()
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model_id = model_id
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def predict(self, seq, image=None):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": seq},
                    {"type": "text", "text": "Please answer YES or NO!"},
                ],
            }
        ]
        if image is not None:
            # Encode the image as a base64 string and add it to the messages list
            if isinstance(image, Image.Image):
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            elif isinstance(image, str):
                with open(image, "rb") as f:
                    b64_image = base64.b64encode(f.read()).decode("utf-8")
            else:
                raise ValueError("Unsupported image type")
            messages[0]["content"].insert(
                0,
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            )
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=512,
            max_completion_tokens=32,
            n=1,
            temperature=0.7,
            logprobs=True,
        )
        content = response.choices[0].message.content
        print(f"Predict: {seq} -> {content}")
        label = torch.Tensor([0 if content[:3].upper() == "YES" else 1]).squeeze().type(torch.int64).to(self.device)
        probs = torch.Tensor([0.0, 0.0]).to(self.device)
        return label, probs
