# CHANGELOG

## v202505.01

### 新增

- 初始项目结构，包括 `captcha_adversarial` 主包和 `adv_attacks` 子模块。
- 添加 `vlattack.py`，实现基于 Gradio 的可视化对抗攻击演示界面。
- 支持 BLIP、Qwen2.5-VL、OpenAI 多模态模型的预测与攻击。
- 实现 BSA、BERT、VLA 等多种攻击方式。
- 集成模型加载、攻击流程、候选生成等功能。
- 提供 `requirements.txt` 依赖文件。
- 添加 MIT License 和项目说明文档 `README.md`。

### 目录结构

- `captcha_adversarial/` 主代码目录
- `tmp/` 临时文件与图片
- `var/` 词典等辅助文件

---
