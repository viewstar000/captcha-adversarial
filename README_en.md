Captcha Adversarial
===================

[中文](README.md) | [English](README_en.md)

An experimental project of human-machine CAPTCHA based on adversarial design approach, ONLY for technical exchange.

Design Philosophy
-------------------

[Design Philosophy](docs/design_en.md)

Implemented Features
-------------------

### Generating Adversarial Attack Samples Against Multimodal Language Models

Supported attack algorithms include Bert-Attack, BSA-Attack, VLAttack, etc., with target models supporting BLIP, Qwen2.5-VL, etc.

Usage Method:

```bash
# Download the code
git clone https://github.com/viewstar000/captcha-adversarial.git
cd captcha-adversarial
# Create a virtual execution environment
virtualenv .venv
source .venv/bin/activate
# Install dependencies
pip install -r requirements.txt
# Start the generation tool
python -m captcha_adversarial.adv_attacks.apps.vlattack
```

References:

| Method        | Paper                                                                                                                        | Code                                              |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Bert-Attack   | [BERT-ATTACK: Adversarial Attack Against BERT Using BERT](https://arxiv.org/abs/2004.09984)                                  | [LINK](https://github.com/LinyangLee/BERT-Attack) |
| BSA, VLAttack | [VLATTACK: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models](https://arxiv.org/abs/2310.04655) | [LINK](https://github.com/ericyinyzy/VLAttack)    |
